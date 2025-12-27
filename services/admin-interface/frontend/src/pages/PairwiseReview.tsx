import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { trainingApi, videosApi } from '@/api/client'

interface VideoPair {
  video_id_1: string
  video_id_2: string
  pending_pairs: number
  total_pairs: number
  completed_pairs: number
  status?: string
}

export default function PairwiseReview() {
  const navigate = useNavigate()
  const [pair, setPair] = useState<VideoPair | null>(null)
  const [stats, setStats] = useState<any>(null)
  const [ranking, setRanking] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [selectedWinner, setSelectedWinner] = useState<number | null>(null)
  const [confidence, setConfidence] = useState('confident')
  const [showRanking, setShowRanking] = useState(false)
  
  const video1Ref = useRef<HTMLVideoElement>(null)
  const video2Ref = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    loadNextPair()
    loadStats()
  }, [])

  const loadNextPair = async () => {
    setLoading(true)
    setSelectedWinner(null)
    try {
      const data = await trainingApi.getNextPairwise()
      setPair(data)
    } catch (error) {
      console.error('Failed to load pair:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const [statsData, rankingData] = await Promise.all([
        trainingApi.getPairwiseStats(),
        trainingApi.getPairwiseRanking()
      ])
      setStats(statsData)
      setRanking(rankingData)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const handleSubmit = async () => {
    if (!pair || selectedWinner === null) return

    setSubmitting(true)
    try {
      await trainingApi.submitPairwise(
        pair.video_id_1,
        pair.video_id_2,
        selectedWinner,
        confidence
      )
      await loadStats()
      await loadNextPair()
    } catch (error) {
      console.error('Failed to submit:', error)
      alert('Failed to submit comparison')
    } finally {
      setSubmitting(false)
    }
  }

  const togglePlayback = () => {
    if (video1Ref.current && video2Ref.current) {
      if (isPlaying) {
        video1Ref.current.pause()
        video2Ref.current.pause()
      } else {
        video1Ref.current.play()
        video2Ref.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const restartVideos = () => {
    if (video1Ref.current && video2Ref.current) {
      video1Ref.current.currentTime = 0
      video2Ref.current.currentTime = 0
      video1Ref.current.play()
      video2Ref.current.play()
      setIsPlaying(true)
    }
  }

  // Sync video playback
  useEffect(() => {
    const video1 = video1Ref.current
    const video2 = video2Ref.current
    
    if (!video1 || !video2) return

    const syncPlayback = () => {
      if (Math.abs(video1.currentTime - video2.currentTime) > 0.1) {
        video2.currentTime = video1.currentTime
      }
    }

    video1.addEventListener('timeupdate', syncPlayback)
    return () => video1.removeEventListener('timeupdate', syncPlayback)
  }, [pair])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <div className="text-muted-foreground">Loading video pair...</div>
        </div>
      </div>
    )
  }

  if (pair?.status === 'all_completed') {
    return (
      <div className="text-center py-12">
        <div className="text-6xl mb-4">🎉</div>
        <h2 className="text-3xl font-bold mb-4">All Comparisons Complete!</h2>
        <p className="text-muted-foreground mb-8">
          You've completed all {pair.total_pairs} pairwise comparisons.
        </p>
        <button
          onClick={() => setShowRanking(true)}
          className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
        >
          View Lameness Ranking
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold">Pairwise Comparison</h2>
          <p className="text-muted-foreground mt-1">
            Compare videos to build a lameness hierarchy
          </p>
        </div>
        <div className="flex items-center gap-4">
          {stats && (
            <div className="text-sm text-muted-foreground">
              Progress: {stats.pairs_compared} / {stats.total_possible_pairs} pairs
              ({((stats.pairs_compared / stats.total_possible_pairs) * 100).toFixed(1)}%)
            </div>
          )}
          <button
            onClick={() => setShowRanking(!showRanking)}
            className="px-4 py-2 border rounded-lg hover:bg-accent"
          >
            {showRanking ? 'Hide Ranking' : 'Show Ranking'}
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      {stats && (
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all"
            style={{ width: `${(stats.pairs_compared / stats.total_possible_pairs) * 100}%` }}
          />
        </div>
      )}

      {/* Show Ranking Panel */}
      {showRanking && ranking && (
        <div className="border rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Lameness Ranking (Elo-based)</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Higher Elo = More Lame. Based on {ranking.total_comparisons} comparisons.
          </p>
          <div className="grid gap-2 max-h-64 overflow-y-auto">
            {ranking.ranking.map((item: any) => (
              <div
                key={item.video_id}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 flex items-center justify-center bg-primary text-primary-foreground rounded-full text-sm font-bold">
                    {item.rank}
                  </span>
                  <span className="font-mono text-sm">{item.video_id.slice(0, 8)}...</span>
                </div>
                <div className={`font-medium ${
                  item.elo_rating > 1550 ? 'text-red-600' :
                  item.elo_rating < 1450 ? 'text-green-600' : 'text-gray-600'
                }`}>
                  {item.elo_rating} Elo
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Comparison Area */}
      {pair && (
        <>
          {/* Videos Side by Side */}
          <div className="grid grid-cols-2 gap-6">
            {/* Video A */}
            <div className="space-y-4">
              <div
                className={`border-4 rounded-lg overflow-hidden transition-colors ${
                  selectedWinner === 1 ? 'border-red-500' : 'border-transparent'
                }`}
              >
                <video
                  ref={video1Ref}
                  src={videosApi.getStreamUrl(pair.video_id_1)}
                  className="w-full aspect-video bg-black"
                  loop
                  muted
                />
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold mb-2">Video A</div>
                <button
                  onClick={() => setSelectedWinner(1)}
                  className={`w-full py-3 rounded-lg font-medium transition-colors ${
                    selectedWinner === 1
                      ? 'bg-red-600 text-white'
                      : 'bg-gray-100 hover:bg-gray-200'
                  }`}
                >
                  Video A is More Lame
                </button>
              </div>
            </div>

            {/* Video B */}
            <div className="space-y-4">
              <div
                className={`border-4 rounded-lg overflow-hidden transition-colors ${
                  selectedWinner === 2 ? 'border-red-500' : 'border-transparent'
                }`}
              >
                <video
                  ref={video2Ref}
                  src={videosApi.getStreamUrl(pair.video_id_2)}
                  className="w-full aspect-video bg-black"
                  loop
                  muted
                />
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold mb-2">Video B</div>
                <button
                  onClick={() => setSelectedWinner(2)}
                  className={`w-full py-3 rounded-lg font-medium transition-colors ${
                    selectedWinner === 2
                      ? 'bg-red-600 text-white'
                      : 'bg-gray-100 hover:bg-gray-200'
                  }`}
                >
                  Video B is More Lame
                </button>
              </div>
            </div>
          </div>

          {/* Playback Controls */}
          <div className="flex justify-center gap-4">
            <button
              onClick={restartVideos}
              className="px-6 py-2 border rounded-lg hover:bg-accent"
            >
              ↺ Restart
            </button>
            <button
              onClick={togglePlayback}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
          </div>

          {/* Equal/Can't Decide Option */}
          <div className="text-center">
            <button
              onClick={() => setSelectedWinner(0)}
              className={`px-6 py-2 rounded-lg transition-colors ${
                selectedWinner === 0
                  ? 'bg-gray-600 text-white'
                  : 'border hover:bg-accent'
              }`}
            >
              Equal / Can't Decide
            </button>
          </div>

          {/* Confidence Selection */}
          {selectedWinner !== null && (
            <div className="flex justify-center gap-4">
              <span className="text-sm text-muted-foreground self-center">Confidence:</span>
              {['very_confident', 'confident', 'uncertain'].map((conf) => (
                <button
                  key={conf}
                  onClick={() => setConfidence(conf)}
                  className={`px-4 py-2 rounded-lg text-sm capitalize transition-colors ${
                    confidence === conf
                      ? 'bg-primary text-primary-foreground'
                      : 'border hover:bg-accent'
                  }`}
                >
                  {conf.replace('_', ' ')}
                </button>
              ))}
            </div>
          )}

          {/* Submit Button */}
          <div className="flex justify-center">
            <button
              onClick={handleSubmit}
              disabled={selectedWinner === null || submitting}
              className="px-8 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? 'Submitting...' : 'Submit & Next Pair'}
            </button>
          </div>

          {/* Instructions */}
          <div className="bg-gray-50 rounded-lg p-4 text-sm text-muted-foreground">
            <h4 className="font-semibold mb-2">Instructions</h4>
            <ul className="list-disc list-inside space-y-1">
              <li>Watch both videos carefully, focusing on the cow's gait</li>
              <li>Look for: arched back, head bobbing, uneven stride, favoring a leg</li>
              <li>Select which cow appears MORE lame, or choose "Equal" if you can't tell</li>
              <li>Indicate your confidence level in the assessment</li>
              <li>Your comparisons help build a lameness hierarchy using Elo ranking</li>
            </ul>
          </div>
        </>
      )}

      {/* Keyboard shortcuts hint */}
      <div className="text-center text-xs text-muted-foreground">
        Tip: Use keyboard shortcuts - <kbd className="px-1 bg-gray-100 rounded">A</kbd> for Video A,{' '}
        <kbd className="px-1 bg-gray-100 rounded">B</kbd> for Video B,{' '}
        <kbd className="px-1 bg-gray-100 rounded">E</kbd> for Equal,{' '}
        <kbd className="px-1 bg-gray-100 rounded">Enter</kbd> to submit
      </div>
    </div>
  )
}

