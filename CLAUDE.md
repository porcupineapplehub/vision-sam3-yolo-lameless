# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Cow Lameness Detection ML Pipeline System** - a Docker-based microservices architecture with 22 services for automated dairy cow lameness detection. The system processes video uploads through computer vision (YOLO, SAM3, DINOv3), pose estimation (T-LEAP), deep learning (TCN, Transformer, Graph Transformer), and traditional ML (XGBoost/CatBoost/LightGBM ensemble), then fuses predictions with human-in-the-loop comparisons.

## Build and Run Commands

### Full Deployment
```bash
./scripts/deploy.sh              # Fresh deployment
./scripts/deploy.sh --clean      # Clean start (removes data/volumes)
./scripts/deploy.sh --skip-build # Skip image rebuilding
```

### Docker Compose
```bash
docker compose up -d                      # Start all services
docker compose build <service-name>       # Build single service
docker compose up -d <service-name>       # Restart single service
docker compose logs <service-name>        # View service logs
```

### Database Initialization
```bash
docker compose exec postgres psql -U lameness_user -d lameness_db < scripts/init_db.sql
```

### Local Development (without Docker)
```bash
# Start infrastructure only
docker compose up -d postgres nats qdrant

# Run backend locally
cd services/admin-interface/backend
conda env create -f environment.yml
conda activate admin-backend
python -m uvicorn app.main:app --reload --port 8000

# Run frontend locally
cd services/admin-interface/frontend
npm ci
npm run dev
```

## Testing and Linting

### Backend (Python)
```bash
# Linting
flake8 services/admin-interface/backend --max-line-length=120 --ignore=E501,W503
black --check services/admin-interface/backend
isort --check-only services/admin-interface/backend

# Tests
cd services/admin-interface/backend
pytest --cov=app --cov-report=xml
```

### Frontend (TypeScript/React)
```bash
cd services/admin-interface/frontend
npm run lint
npm test -- --passWithNoTests
```

### Terraform
```bash
cd terraform
terraform fmt -check -recursive
terraform init -backend=false
terraform validate
```

## Architecture

### Event-Driven Microservices
All 22 services communicate via **NATS message broker**. The canonical configuration for NATS subjects and data paths is in `shared/config/config.yaml`.

### Data Flow
```
Video Upload → video-ingestion → video-preprocessing → clip-curation (5s canonical clips)
                                                              ↓
              ┌──────────────────────────────────────────────────────────────┐
              ↓                    ↓                    ↓                    ↓
         yolo-pipeline      sam3-pipeline       dinov3-pipeline       tleap-pipeline
              ↓                    ↓                    ↓                    ↓
              └──────────────────────────────────────────────────────────────┘
                                              ↓
              ┌──────────────────────────────────────────────────────────────┐
              ↓                    ↓                    ↓                    ↓
         tcn-pipeline    transformer-pipeline    gnn-pipeline    graph-transformer-pipeline
              ↓                    ↓                    ↓                    ↓
              └──────────────────────────────────────────────────────────────┘
                                              ↓
                                        ml-pipeline
                                              ↓
                                       fusion-service → analysis.complete
```

### Key NATS Subjects
- `video.uploaded` → `video.preprocessed` → `video.curated`
- `pipeline.{yolo,sam3,dinov3,tleap}` - Feature extraction results
- `pipeline.{tcn,transformer,gnn,graph_transformer,ml}` - Model predictions
- `pipeline.fusion` - Final fused prediction
- `tracking.complete` - Cow identity tracking results
- `analysis.complete` - End-to-end processing complete

### Data Storage Pattern
Services write results to `data/results/<pipeline-name>/` AND publish NATS messages with `results_path` and `features`.

### Service Entry Points
- Admin Frontend: React+TypeScript at `services/admin-interface/frontend`
- Admin Backend: FastAPI at `services/admin-interface/backend/app/main.py`
- ML Pipelines: Each in `services/<pipeline-name>/` with own Dockerfile and environment.yml

### Key Infrastructure
| Service | Port | Purpose |
|---------|------|---------|
| admin-frontend | 3000 | React UI |
| admin-backend | 8000 | FastAPI REST API |
| PostgreSQL | 5432 | Metadata, user auth, cow registry |
| NATS | 4222 | Message broker |
| Qdrant | 6333 | Vector DB for DINOv3 embeddings |

### Tech Stack
- **Backend**: Python 3.10, FastAPI, SQLAlchemy, NATS, PyTorch
- **Frontend**: React 18, TypeScript, Vite, TailwindCSS, Radix UI
- **ML**: YOLO (Ultralytics), SAM3, DINOv3, XGBoost, CatBoost, LightGBM, PyTorch Geometric
- **Infrastructure**: Docker Compose, PostgreSQL 15, NATS with JetStream, Qdrant, Terraform (AWS)

## Key Files

- `shared/config/config.yaml` - Canonical config for NATS subjects, database URLs, data paths, model settings
- `env.example` - Environment variable template (copy to `.env`)
- `scripts/init_db.sql` - Database schema
- `docker-compose.yml` - All 22 service definitions
- `docs/ARCHITECTURE.md` - Detailed architecture diagrams
- `docs/PIPELINES_DETAILED.md` - ML pipeline walkthrough

## Default Credentials

- **Admin**: admin@example.com / adminpass123
- **Researcher**: researcher@example.com / researcher123
- **Rater**: rater@example.com / rater123
