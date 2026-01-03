"""
Cow Re-Identification Matcher using Qdrant Vector Database.

Provides cross-video cow identification by matching appearance embeddings
against a database of known cow identities.
"""
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from uuid import UUID, uuid4
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct


@dataclass
class CowIdentity:
    """Represents a known cow identity"""
    identity_id: UUID
    cow_id: str  # Human-readable ID (e.g., "COW-001")
    tag_number: Optional[str] = None
    total_sightings: int = 0
    embedding: Optional[np.ndarray] = None
    embedding_dim: int = 768


@dataclass
class ReIDMatch:
    """Result of a Re-ID query"""
    identity_id: UUID
    cow_id: str
    similarity: float
    confidence: str  # "high", "medium", "low"
    is_new_identity: bool = False


class CowReIDMatcher:
    """
    Cow Re-Identification using vector similarity search.

    Uses DINOv3 embeddings stored in Qdrant to identify cows across
    videos and time.

    Features:
    - Fast similarity search using Qdrant
    - Momentum-based embedding updates for known cows
    - Automatic new identity creation
    - Confidence estimation based on similarity
    """

    COLLECTION_NAME = "cow_identities"
    SIMILARITY_THRESHOLD_HIGH = 0.85  # High confidence match
    SIMILARITY_THRESHOLD_MEDIUM = 0.75  # Medium confidence match
    SIMILARITY_THRESHOLD_LOW = 0.65  # Low confidence / possible match

    def __init__(
        self,
        qdrant_url: str = "http://qdrant:6333",
        embedding_dim: int = 768,
        auto_create_identities: bool = True,
        embedding_momentum: float = 0.9
    ):
        """
        Initialize Re-ID matcher.

        Args:
            qdrant_url: Qdrant server URL
            embedding_dim: Dimension of embeddings (768 for DINOv3)
            auto_create_identities: Whether to create new identities for unmatched cows
            embedding_momentum: Momentum for updating known cow embeddings
        """
        self.qdrant_url = qdrant_url
        self.embedding_dim = embedding_dim
        self.auto_create_identities = auto_create_identities
        self.embedding_momentum = embedding_momentum

        self.client: Optional[QdrantClient] = None
        self.identity_counter = 0

    async def connect(self):
        """Connect to Qdrant and ensure collection exists"""
        self.client = QdrantClient(url=self.qdrant_url)

        # Create collection if not exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created Qdrant collection: {self.COLLECTION_NAME}")
        else:
            print(f"Using existing Qdrant collection: {self.COLLECTION_NAME}")

        # Get current count for ID generation
        collection_info = self.client.get_collection(self.COLLECTION_NAME)
        self.identity_counter = collection_info.points_count

    def match_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[Optional[ReIDMatch], List[ReIDMatch]]:
        """
        Match an embedding against known cow identities.

        Args:
            embedding: (D,) embedding vector from DINOv3
            top_k: Number of candidates to return

        Returns:
            best_match: Best matching identity or None
            candidates: List of potential matches
        """
        if self.client is None:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Search for similar embeddings using query_points API
        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=embedding.tolist(),
            limit=top_k,
            with_payload=True
        )

        candidates = []
        for point in response.points:
            confidence = self._score_to_confidence(point.score)
            candidates.append(ReIDMatch(
                identity_id=UUID(point.payload["identity_id"]),
                cow_id=point.payload["cow_id"],
                similarity=point.score,
                confidence=confidence
            ))

        # Determine best match
        best_match = None
        if candidates and candidates[0].similarity >= self.SIMILARITY_THRESHOLD_LOW:
            best_match = candidates[0]

        return best_match, candidates

    def match_or_create(
        self,
        embedding: np.ndarray,
        video_id: str,
        track_id: int,
        metadata: Optional[Dict] = None
    ) -> ReIDMatch:
        """
        Match embedding or create new identity.

        This is the main entry point for Re-ID during tracking.

        Args:
            embedding: (D,) embedding vector
            video_id: Source video ID
            track_id: Track ID within video
            metadata: Additional metadata

        Returns:
            ReIDMatch with matched or new identity
        """
        best_match, candidates = self.match_embedding(embedding)

        if best_match is not None and best_match.similarity >= self.SIMILARITY_THRESHOLD_MEDIUM:
            # Update existing identity with new embedding
            self._update_identity_embedding(best_match.identity_id, embedding)
            return best_match

        if self.auto_create_identities:
            # Create new identity
            new_identity = self.create_identity(
                embedding=embedding,
                tag_number=None,
                metadata={"first_video": video_id, "first_track": track_id, **(metadata or {})}
            )
            return ReIDMatch(
                identity_id=new_identity.identity_id,
                cow_id=new_identity.cow_id,
                similarity=1.0,
                confidence="high",
                is_new_identity=True
            )

        # No match and auto-create disabled
        return ReIDMatch(
            identity_id=uuid4(),
            cow_id="UNKNOWN",
            similarity=candidates[0].similarity if candidates else 0.0,
            confidence="low",
            is_new_identity=True
        )

    def create_identity(
        self,
        embedding: np.ndarray,
        tag_number: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> CowIdentity:
        """
        Create a new cow identity.

        Args:
            embedding: Initial embedding vector
            tag_number: Optional farm tag number
            metadata: Additional metadata

        Returns:
            New CowIdentity object
        """
        if self.client is None:
            raise RuntimeError("Not connected to Qdrant. Call connect() first.")

        self.identity_counter += 1
        identity_id = uuid4()
        cow_id = f"COW-{self.identity_counter:04d}"

        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Create point in Qdrant
        point = PointStruct(
            id=str(identity_id),
            vector=embedding.tolist(),
            payload={
                "identity_id": str(identity_id),
                "cow_id": cow_id,
                "tag_number": tag_number,
                "total_sightings": 1,
                **(metadata or {})
            }
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point]
        )

        return CowIdentity(
            identity_id=identity_id,
            cow_id=cow_id,
            tag_number=tag_number,
            total_sightings=1,
            embedding=embedding,
            embedding_dim=len(embedding)
        )

    def _update_identity_embedding(self, identity_id: UUID, new_embedding: np.ndarray):
        """
        Update identity embedding using momentum smoothing.

        embedding_new = momentum * embedding_old + (1 - momentum) * embedding
        """
        if self.client is None:
            return

        # Get current embedding
        points = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[str(identity_id)],
            with_vectors=True
        )

        if not points:
            return

        point = points[0]
        old_embedding = np.array(point.vector)
        new_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)

        # Momentum update
        updated_embedding = (
            self.embedding_momentum * old_embedding +
            (1 - self.embedding_momentum) * new_embedding
        )
        updated_embedding = updated_embedding / (np.linalg.norm(updated_embedding) + 1e-8)

        # Update sightings count
        sightings = point.payload.get("total_sightings", 0) + 1

        # Update in Qdrant
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[PointStruct(
                id=str(identity_id),
                vector=updated_embedding.tolist(),
                payload={
                    **point.payload,
                    "total_sightings": sightings
                }
            )]
        )

    def _score_to_confidence(self, score: float) -> str:
        """Convert similarity score to confidence level"""
        if score >= self.SIMILARITY_THRESHOLD_HIGH:
            return "high"
        elif score >= self.SIMILARITY_THRESHOLD_MEDIUM:
            return "medium"
        elif score >= self.SIMILARITY_THRESHOLD_LOW:
            return "low"
        return "none"

    def get_identity(self, identity_id: UUID) -> Optional[CowIdentity]:
        """Get identity by ID"""
        if self.client is None:
            return None

        points = self.client.retrieve(
            collection_name=self.COLLECTION_NAME,
            ids=[str(identity_id)],
            with_vectors=True
        )

        if not points:
            return None

        point = points[0]
        return CowIdentity(
            identity_id=UUID(point.payload["identity_id"]),
            cow_id=point.payload["cow_id"],
            tag_number=point.payload.get("tag_number"),
            total_sightings=point.payload.get("total_sightings", 0),
            embedding=np.array(point.vector) if point.vector else None,
            embedding_dim=self.embedding_dim
        )

    def get_all_identities(self, limit: int = 100) -> List[CowIdentity]:
        """Get all known identities"""
        if self.client is None:
            return []

        results = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            limit=limit,
            with_vectors=False
        )

        identities = []
        for point in results[0]:
            identities.append(CowIdentity(
                identity_id=UUID(point.payload["identity_id"]),
                cow_id=point.payload["cow_id"],
                tag_number=point.payload.get("tag_number"),
                total_sightings=point.payload.get("total_sightings", 0),
                embedding_dim=self.embedding_dim
            ))

        return identities

    def get_statistics(self) -> dict:
        """Get Re-ID statistics"""
        if self.client is None:
            return {"status": "disconnected"}

        collection_info = self.client.get_collection(self.COLLECTION_NAME)
        return {
            "status": "connected",
            "collection": self.COLLECTION_NAME,
            "total_identities": collection_info.points_count,
            "embedding_dim": self.embedding_dim,
            "similarity_threshold": self.SIMILARITY_THRESHOLD_MEDIUM
        }
