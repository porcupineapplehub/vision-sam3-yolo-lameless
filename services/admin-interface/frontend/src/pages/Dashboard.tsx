import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { videosApi, trainingApi } from '@/api/client'

export default function Dashboard() {
  const [videos, setVideos] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [videoData, statsData] = await Promise.all([
        videosApi.list(),
        trainingApi.getStats().catch(() => null)
      ])
      setVideos(videoData.videos || [])
      setStats(statsData)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  const labeledCount = videos.filter(v => v.has_label).length
  const soundCount = videos.filter(v => v.label === 0).length
  const lameCount = videos.filter(v => v.label === 1).length

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-bold">Dashboard</h2>
        <Link
          to="/upload"
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
        >
          Upload Videos
        </Link>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="p-4 border rounded-lg bg-background">
          <p className="text-sm text-muted-foreground">Total Videos</p>
          <p className="text-2xl font-bold">{videos.length}</p>
        </div>
        <div className="p-4 border rounded-lg bg-background">
          <p className="text-sm text-muted-foreground">Labeled</p>
          <p className="text-2xl font-bold">{labeledCount}</p>
        </div>
        <div className="p-4 border rounded-lg bg-green-50 border-green-200">
          <p className="text-sm text-green-700">Sound</p>
          <p className="text-2xl font-bold text-green-800">{soundCount}</p>
        </div>
        <div className="p-4 border rounded-lg bg-red-50 border-red-200">
          <p className="text-sm text-red-700">Lame</p>
          <p className="text-2xl font-bold text-red-800">{lameCount}</p>
        </div>
      </div>

      {/* Video list */}
      <h3 className="text-xl font-semibold mb-4">Videos</h3>
      <div className="grid gap-2">
        {videos.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No videos uploaded yet. <Link to="/upload" className="text-primary">Upload some now</Link>
          </div>
        ) : (
          videos.map((video) => (
            <Link
              key={video.video_id}
              to={`/video/${video.video_id}`}
              className="block p-4 border rounded-lg hover:bg-accent transition-colors"
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  {/* Label indicator */}
                  {video.has_label && (
                    <span className={`px-2 py-1 text-xs font-medium rounded ${
                      video.label === 0 
                        ? 'bg-green-100 text-green-700' 
                        : 'bg-red-100 text-red-700'
                    }`}>
                      {video.label === 0 ? 'Sound' : 'Lame'}
                    </span>
                  )}
                  <div>
                    <h3 className="font-semibold">{video.filename}</h3>
                    <p className="text-sm text-muted-foreground">
                      {video.has_analysis ? '✓ Analyzed' : 'Pending analysis'}
                      {!video.has_label && ' • Unlabeled'}
                    </p>
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  {(video.file_size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  )
}

