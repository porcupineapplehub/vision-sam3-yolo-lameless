/**
 * Pipeline Analysis Page
 * Comprehensive deep-dive into pipeline results for researchers
 * Features: Video player with overlays, 12 pipeline tabs, timeline scrubber, export
 */
import { useEffect, useState, useRef, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Activity,
  Brain,
  Eye,
  Network,
  Cpu,
  GitMerge,
  CheckCircle,
  XCircle,
  AlertCircle,
  ArrowLeft,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Download,
  Loader2,
  BarChart3,
  LineChart,
  PieChart,
  Layers,
  Zap,
  Share2,
  Info,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  FileJson,
  FileText,
  Box,
  Target,
  Sparkles,
  TrendingUp
} from 'lucide-react'
import { videosApi, pipelineResultsApi, shapApi } from '../api/client'

const API_BASE = '/api'

// Pipeline configurations with icons and descriptions
const PIPELINE_CONFIG = {
  summary: { name: 'Summary', icon: BarChart3, color: 'blue', description: 'Overview of all pipeline results' },
  yolo: { name: 'YOLO', icon: Target, color: 'yellow', description: 'Object detection (cow bounding boxes)' },
  sam3: { name: 'SAM3', icon: Layers, color: 'purple', description: 'Instance segmentation masks' },
  dinov3: { name: 'DINOv3', icon: Eye, color: 'green', description: 'Visual embeddings & similarity' },
  tleap: { name: 'T-LEAP', icon: Activity, color: 'orange', description: 'Pose estimation & locomotion' },
  tcn: { name: 'TCN', icon: LineChart, color: 'cyan', description: 'Temporal convolutional network' },
  transformer: { name: 'Transformer', icon: Zap, color: 'pink', description: 'Self-attention gait analysis' },
  gnn: { name: 'GNN', icon: Share2, color: 'indigo', description: 'Graph neural network (GraphGPS)' },
  graph_transformer: { name: 'GraphT', icon: Sparkles, color: 'amber', description: 'Graph Transformer (Graphormer)' },
  ml: { name: 'ML Ensemble', icon: Brain, color: 'red', description: 'CatBoost, XGBoost, LightGBM' },
  fusion: { name: 'Fusion', icon: GitMerge, color: 'emerald', description: 'Final ensemble prediction' },
  shap: { name: 'SHAP', icon: PieChart, color: 'violet', description: 'Feature importance & explainability' }
}

type PipelineKey = keyof typeof PIPELINE_CONFIG

interface PipelineResult {
  status: 'success' | 'error' | 'pending' | 'not_available'
  data: any
}

interface AllResults {
  [key: string]: PipelineResult
}

export default function PipelineAnalysis() {
  const { videoId } = useParams<{ videoId: string }>()
  const navigate = useNavigate()

  // State
  const [loading, setLoading] = useState(true)
  const [videoInfo, setVideoInfo] = useState<any>(null)
  const [results, setResults] = useState<AllResults>({})
  const [activeTab, setActiveTab] = useState<PipelineKey>('summary')
  const [currentFrame, setCurrentFrame] = useState(0)
  const [totalFrames, setTotalFrames] = useState(100)
  const [isPlaying, setIsPlaying] = useState(false)
  const [overlays, setOverlays] = useState({ detections: true, pose: true, segmentation: false })
  const [error, setError] = useState<string | null>(null)
  const [frameData, setFrameData] = useState<any>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const animationRef = useRef<number>()

  // Load all data on mount
  useEffect(() => {
    if (videoId) {
      loadAllData()
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [videoId])

  // Update frame data when current frame changes
  useEffect(() => {
    if (videoId && currentFrame > 0) {
      loadFrameData(currentFrame)
    }
  }, [videoId, currentFrame])

  const loadAllData = async () => {
    setLoading(true)
    setError(null)

    try {
      // Load video info
      const videoResponse = await fetch(`${API_BASE}/videos/${videoId}`)
      if (videoResponse.ok) {
        const info = await videoResponse.json()
        setVideoInfo(info)
        setTotalFrames(info.frame_count || 100)
      }

      // Load all pipeline results
      const allResponse = await fetch(`${API_BASE}/analysis/${videoId}/all`)
      if (allResponse.ok) {
        const data = await allResponse.json()
        setResults(data.pipelines || {})
      }

      // Load SHAP data separately
      try {
        const shapResponse = await fetch(`${API_BASE}/shap/${videoId}/local`)
        if (shapResponse.ok) {
          const shapData = await shapResponse.json()
          setResults(prev => ({
            ...prev,
            shap: { status: 'success', data: shapData }
          }))
        }
      } catch (e) {
        // SHAP may not be available
      }
    } catch (err) {
      setError('Failed to load analysis data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadFrameData = async (frame: number) => {
    try {
      const response = await fetch(`${API_BASE}/analysis/${videoId}/frames/${frame}`)
      if (response.ok) {
        setFrameData(await response.json())
      }
    } catch (e) {
      console.error('Failed to load frame data:', e)
    }
  }

  // Video controls
  const handleVideoTimeUpdate = useCallback(() => {
    if (videoRef.current && videoInfo?.fps) {
      const frame = Math.floor(videoRef.current.currentTime * videoInfo.fps)
      setCurrentFrame(frame)
    }
  }, [videoInfo?.fps])

  const seekToFrame = (frame: number) => {
    if (videoRef.current && videoInfo?.fps) {
      videoRef.current.currentTime = frame / videoInfo.fps
      setCurrentFrame(frame)
    }
  }

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const stepFrame = (delta: number) => {
    seekToFrame(Math.max(0, Math.min(totalFrames - 1, currentFrame + delta)))
  }

  // Export handlers
  const handleExport = (format: 'json' | 'csv') => {
    const url = pipelineResultsApi.exportResults(videoId!, format)
    window.open(url, '_blank')
  }

  // Render status icon
  const StatusIcon = ({ status }: { status: string }) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'pending':
        return <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />
      default:
        return <AlertCircle className="w-4 h-4 text-gray-400" />
    }
  }

  // Get pipeline color class
  const getColorClass = (color: string, type: 'bg' | 'border' | 'text') => {
    const colors: Record<string, Record<string, string>> = {
      blue: { bg: 'bg-blue-100', border: 'border-blue-500', text: 'text-blue-600' },
      yellow: { bg: 'bg-yellow-100', border: 'border-yellow-500', text: 'text-yellow-600' },
      purple: { bg: 'bg-purple-100', border: 'border-purple-500', text: 'text-purple-600' },
      green: { bg: 'bg-green-100', border: 'border-green-500', text: 'text-green-600' },
      orange: { bg: 'bg-orange-100', border: 'border-orange-500', text: 'text-orange-600' },
      cyan: { bg: 'bg-cyan-100', border: 'border-cyan-500', text: 'text-cyan-600' },
      pink: { bg: 'bg-pink-100', border: 'border-pink-500', text: 'text-pink-600' },
      indigo: { bg: 'bg-indigo-100', border: 'border-indigo-500', text: 'text-indigo-600' },
      amber: { bg: 'bg-amber-100', border: 'border-amber-500', text: 'text-amber-600' },
      red: { bg: 'bg-red-100', border: 'border-red-500', text: 'text-red-600' },
      emerald: { bg: 'bg-emerald-100', border: 'border-emerald-500', text: 'text-emerald-600' },
      violet: { bg: 'bg-violet-100', border: 'border-violet-500', text: 'text-violet-600' },
    }
    return colors[color]?.[type] || 'bg-gray-100'
  }

  // Count pipeline statuses for stats bar
  const pipelineKeys = ['yolo', 'sam3', 'dinov3', 'tleap', 'tcn', 'transformer', 'gnn', 'graph_transformer', 'ml', 'fusion']
  const successCount = pipelineKeys.filter(k => results[k]?.status === 'success').length
  const errorCount = pipelineKeys.filter(k => results[k]?.status === 'error').length
  const pendingCount = pipelineKeys.filter(k => results[k]?.status === 'pending').length

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-gray-600 font-medium">Analyzing video pipelines...</p>
          <p className="text-sm text-gray-400 mt-1">Loading results from all ML models</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center p-8 bg-red-50 rounded-2xl border-2 border-red-200 max-w-md">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-red-800 mb-2">Analysis Failed</h2>
          <p className="text-red-600 mb-6">{error}</p>
          <button
            onClick={() => navigate(-1)}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-all font-medium"
          >
            Go Back
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header - Pipeline Monitor style */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Pipeline Analysis</h1>
            <p className="text-sm text-gray-500">
              Deep dive into ML pipeline results for video <span className="font-mono text-gray-700">{videoId?.substring(0, 8)}...</span>
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* Live Status Indicator */}
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${successCount > 0 ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            <span className="text-gray-500">
              {successCount}/{pipelineKeys.length} pipelines
            </span>
          </div>
          <button
            onClick={loadAllData}
            className="flex items-center gap-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
          <div className="relative group">
            <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium">
              <Download className="w-4 h-4" />
              Export
            </button>
            <div className="absolute right-0 mt-2 w-44 bg-white rounded-lg shadow-xl border hidden group-hover:block z-20">
              <button
                onClick={() => handleExport('json')}
                className="w-full px-4 py-3 text-left hover:bg-gray-50 flex items-center gap-3 rounded-t-lg transition-colors"
              >
                <FileJson className="w-4 h-4 text-blue-600" />
                <span className="font-medium">Export JSON</span>
              </button>
              <button
                onClick={() => handleExport('csv')}
                className="w-full px-4 py-3 text-left hover:bg-gray-50 flex items-center gap-3 rounded-b-lg transition-colors border-t"
              >
                <FileText className="w-4 h-4 text-green-600" />
                <span className="font-medium">Export CSV</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Summary Bar - Pipeline Monitor style */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border-2 border-green-200 hover:border-green-300 transition-colors">
          <div className="flex items-center gap-2 text-green-600 mb-1">
            <CheckCircle className="h-4 w-4" />
            <span className="text-sm font-medium">Completed</span>
          </div>
          <p className="text-2xl font-bold text-green-700">{successCount}</p>
        </div>
        <div className="bg-white p-4 rounded-lg border-2 border-red-200 hover:border-red-300 transition-colors">
          <div className="flex items-center gap-2 text-red-600 mb-1">
            <XCircle className="h-4 w-4" />
            <span className="text-sm font-medium">Errors</span>
          </div>
          <p className="text-2xl font-bold text-red-700">{errorCount}</p>
        </div>
        <div className="bg-white p-4 rounded-lg border-2 border-yellow-200 hover:border-yellow-300 transition-colors">
          <div className="flex items-center gap-2 text-yellow-600 mb-1">
            <Loader2 className="h-4 w-4" />
            <span className="text-sm font-medium">Pending</span>
          </div>
          <p className="text-2xl font-bold text-yellow-700">{pendingCount}</p>
        </div>
        <div className="bg-white p-4 rounded-lg border-2 border-blue-200 hover:border-blue-300 transition-colors">
          <div className="flex items-center gap-2 text-blue-600 mb-1">
            <Activity className="h-4 w-4" />
            <span className="text-sm font-medium">Final Score</span>
          </div>
          <p className="text-2xl font-bold text-blue-700">
            {results.fusion?.data?.fusion_result?.final_probability
              ? `${(results.fusion.data.fusion_result.final_probability * 100).toFixed(0)}%`
              : '-'}
          </p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Video Player Section */}
        <div className="col-span-5 space-y-4">
          <div className="bg-white rounded-lg border-2 border-gray-200 hover:border-gray-300 transition-colors overflow-hidden">
            {/* Video Header */}
            <div className="px-4 py-3 border-b bg-gray-50 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Box className="w-4 h-4 text-gray-500" />
                <span className="font-medium text-sm">Video Preview</span>
              </div>
              <span className="text-xs text-gray-500 font-mono">{videoInfo?.fps || 30} fps</span>
            </div>

            <div className="p-4">
              <div className="relative bg-gray-900 rounded-lg overflow-hidden aspect-video">
                <video
                  ref={videoRef}
                  src={videosApi.getStreamUrl(videoId!)}
                  className="w-full h-full object-contain"
                  onTimeUpdate={handleVideoTimeUpdate}
                  onEnded={() => setIsPlaying(false)}
                />
                {/* Overlay indicators */}
                {overlays.detections && frameData?.detections?.length > 0 && (
                  <div className="absolute top-2 left-2 bg-yellow-500/90 text-white text-xs px-2 py-1 rounded-full font-medium">
                    {frameData.detections.length} detections
                  </div>
                )}
                {/* Frame badge */}
                <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-mono">
                  {currentFrame} / {totalFrames}
                </div>
              </div>

              {/* Video Controls */}
              <div className="mt-4 space-y-3">
                <div className="flex items-center justify-center gap-2">
                  <button onClick={() => stepFrame(-10)} className="p-2 hover:bg-gray-100 rounded-lg transition-colors" title="Back 10 frames">
                    <SkipBack className="w-4 h-4" />
                  </button>
                  <button onClick={() => stepFrame(-1)} className="p-2 hover:bg-gray-100 rounded-lg transition-colors" title="Back 1 frame">
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    onClick={togglePlay}
                    className="p-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-all shadow-lg hover:shadow-xl"
                  >
                    {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
                  </button>
                  <button onClick={() => stepFrame(1)} className="p-2 hover:bg-gray-100 rounded-lg transition-colors" title="Forward 1 frame">
                    <ChevronRight className="w-4 h-4" />
                  </button>
                  <button onClick={() => stepFrame(10)} className="p-2 hover:bg-gray-100 rounded-lg transition-colors" title="Forward 10 frames">
                    <SkipForward className="w-4 h-4" />
                  </button>
                </div>

                {/* Frame Scrubber */}
                <div className="space-y-1 px-1">
                  <input
                    type="range"
                    min={0}
                    max={totalFrames - 1}
                    value={currentFrame}
                    onChange={(e) => seekToFrame(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                </div>

                {/* Overlay Toggles */}
                <div className="flex items-center justify-center gap-4 text-sm">
                  <label className="flex items-center gap-2 cursor-pointer hover:text-blue-600 transition-colors">
                    <input
                      type="checkbox"
                      checked={overlays.detections}
                      onChange={(e) => setOverlays({ ...overlays, detections: e.target.checked })}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span>Detections</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer hover:text-orange-600 transition-colors">
                    <input
                      type="checkbox"
                      checked={overlays.pose}
                      onChange={(e) => setOverlays({ ...overlays, pose: e.target.checked })}
                      className="rounded border-gray-300 text-orange-600 focus:ring-orange-500"
                    />
                    <span>Pose</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer hover:text-purple-600 transition-colors">
                    <input
                      type="checkbox"
                      checked={overlays.segmentation}
                      onChange={(e) => setOverlays({ ...overlays, segmentation: e.target.checked })}
                      className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                    />
                    <span>Mask</span>
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Final Prediction Card */}
          {results.fusion?.status === 'success' && results.fusion.data && (
            <div className={`rounded-lg border-2 overflow-hidden transition-all ${
              results.fusion.data.fusion_result?.final_prediction === 1
                ? 'border-red-300 bg-gradient-to-br from-red-50 to-red-100/50'
                : 'border-green-300 bg-gradient-to-br from-green-50 to-green-100/50'
            }`}>
              <div className={`px-4 py-2 border-b ${
                results.fusion.data.fusion_result?.final_prediction === 1
                  ? 'bg-red-100/80 border-red-200'
                  : 'bg-green-100/80 border-green-200'
              }`}>
                <h3 className={`font-semibold text-sm flex items-center gap-2 ${
                  results.fusion.data.fusion_result?.final_prediction === 1
                    ? 'text-red-800'
                    : 'text-green-800'
                }`}>
                  <TrendingUp className="w-4 h-4" />
                  Final Prediction
                </h3>
              </div>
              <div className="p-4">
                <div className="flex items-center justify-between">
                  <div className={`text-3xl font-bold ${
                    results.fusion.data.fusion_result?.final_prediction === 1
                      ? 'text-red-600'
                      : 'text-green-600'
                  }`}>
                    {results.fusion.data.fusion_result?.final_prediction === 1 ? 'LAME' : 'HEALTHY'}
                  </div>
                  <div className="text-right">
                    <div className={`text-2xl font-bold ${
                      results.fusion.data.fusion_result?.final_prediction === 1
                        ? 'text-red-600'
                        : 'text-green-600'
                    }`}>
                      {((results.fusion.data.fusion_result?.final_probability || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">
                      Confidence: <span className="font-medium">{((results.fusion.data.fusion_result?.confidence || 0) * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
                <div className="mt-3 h-3 bg-white/80 rounded-full overflow-hidden border">
                  <div
                    className={`h-full transition-all duration-500 ${
                      results.fusion.data.fusion_result?.final_prediction === 1
                        ? 'bg-gradient-to-r from-red-400 to-red-500'
                        : 'bg-gradient-to-r from-green-400 to-green-500'
                    }`}
                    style={{ width: `${(results.fusion.data.fusion_result?.final_probability || 0) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Pipeline Tabs Section */}
        <div className="col-span-7">
          <div className="bg-white rounded-lg border-2 border-gray-200 hover:border-gray-300 transition-colors overflow-hidden">
            {/* Tab Header */}
            <div className="px-4 py-3 border-b bg-gradient-to-r from-gray-50 to-white flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-blue-600" />
                <span className="font-medium text-sm">Pipeline Results</span>
              </div>
              <span className="text-xs text-gray-500">{successCount} of {pipelineKeys.length} complete</span>
            </div>

            {/* Tab Navigation - Compact Icon Grid */}
            <div className="border-b p-3 bg-gray-50/50">
              <div className="flex items-center gap-3">
                {/* Current Tab Label */}
                  <div className="flex items-center gap-2 min-w-0">
                    {(() => {
                      const config = PIPELINE_CONFIG[activeTab]
                      const Icon = config.icon
                      return (
                        <>
                          <Icon className={`w-5 h-5 ${getColorClass(config.color, 'text')}`} />
                          <span className="font-medium text-gray-800">{config.name}</span>
                        </>
                      )
                    })()}
                  </div>
                  <div className="h-6 w-px bg-gray-200" />
                  {/* Icon Tab Buttons */}
                  <div className="flex flex-wrap gap-1">
                    {(Object.keys(PIPELINE_CONFIG) as PipelineKey[]).map((key) => {
                      const config = PIPELINE_CONFIG[key]
                      const result = results[key]
                      const Icon = config.icon
                      const isActive = activeTab === key

                      return (
                        <button
                          key={key}
                          onClick={() => setActiveTab(key)}
                          title={`${config.name}: ${config.description}`}
                          className={`relative p-2 rounded-lg transition-all ${
                            isActive
                              ? `${getColorClass(config.color, 'bg')} ${getColorClass(config.color, 'text')} ring-2 ring-offset-1 ring-${config.color}-400`
                              : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                          }`}
                        >
                          <Icon className="w-4 h-4" />
                          {/* Status indicator dot */}
                          {key !== 'summary' && result && (
                            <span className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full ${
                              result.status === 'success' ? 'bg-green-500' :
                              result.status === 'error' ? 'bg-red-500' :
                              result.status === 'pending' ? 'bg-yellow-500' : 'bg-gray-300'
                            }`} />
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
                {/* Tab Description */}
                <p className="text-xs text-gray-500 mt-2">
                  {PIPELINE_CONFIG[activeTab].description}
                </p>
              </div>

              {/* Tab Content */}
              <div className="p-6 min-h-[500px]">
                {activeTab === 'summary' && (
                  <SummaryTab results={results} onTabChange={setActiveTab} />
                )}
                {activeTab === 'yolo' && (
                  <YoloTab data={results.yolo?.data} />
                )}
                {activeTab === 'sam3' && (
                  <Sam3Tab data={results.sam3?.data} />
                )}
                {activeTab === 'dinov3' && (
                  <DinOv3Tab data={results.dinov3?.data} videoId={videoId!} />
                )}
                {activeTab === 'tleap' && (
                  <TleapTab data={results.tleap?.data} />
                )}
                {activeTab === 'tcn' && (
                  <TcnTab data={results.tcn?.data} />
                )}
                {activeTab === 'transformer' && (
                  <TransformerTab data={results.transformer?.data} />
                )}
                {activeTab === 'gnn' && (
                  <GnnTab data={results.gnn?.data} />
                )}
                {activeTab === 'graph_transformer' && (
                  <GraphTransformerTab data={results.graph_transformer?.data} />
                )}
                {activeTab === 'ml' && (
                  <MlEnsembleTab data={results.ml?.data} />
                )}
                {activeTab === 'fusion' && (
                  <FusionTab data={results.fusion?.data} />
                )}
                {activeTab === 'shap' && (
                  <ShapTab data={results.shap?.data} />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ============== TAB COMPONENTS ==============

// Summary Tab - Overview of all pipelines
function SummaryTab({ results, onTabChange }: { results: AllResults; onTabChange: (tab: PipelineKey) => void }) {
  const pipelines = ['yolo', 'sam3', 'dinov3', 'tleap', 'tcn', 'transformer', 'gnn', 'graph_transformer', 'ml', 'fusion']

  const getMetricDisplay = (key: string, data: any) => {
    if (!data) return { value: '-', label: 'N/A' }

    switch (key) {
      case 'yolo':
        return { value: data.features?.num_detections || 0, label: 'detections' }
      case 'sam3':
        return { value: `${((data.features?.avg_area_ratio || 0) * 100).toFixed(0)}%`, label: 'coverage' }
      case 'dinov3':
        return { value: data.similar_cases?.length || 0, label: 'similar' }
      case 'tleap':
        return { value: (data.locomotion_features?.lameness_score || 0).toFixed(2), label: 'LS score' }
      case 'tcn':
        return { value: (data.severity_score || 0).toFixed(2), label: 'severity' }
      case 'transformer':
        return { value: (data.severity_score || 0).toFixed(2), label: 'severity' }
      case 'gnn':
        return { value: (data.severity_score || 0).toFixed(2), label: 'severity' }
      case 'graph_transformer':
        return { value: (data.graph_prediction || 0).toFixed(2), label: 'graph pred' }
      case 'ml':
        return { value: (data.predictions?.ensemble?.probability || 0).toFixed(2), label: 'prob' }
      case 'fusion':
        return { value: (data.fusion_result?.final_probability || 0).toFixed(2), label: 'final' }
      default:
        return { value: '-', label: '' }
    }
  }

  const getStatusBorderColor = (status: string, color: string) => {
    if (status === 'success') {
      const borderColors: Record<string, string> = {
        yellow: 'border-yellow-300',
        purple: 'border-purple-300',
        green: 'border-green-300',
        orange: 'border-orange-300',
        cyan: 'border-cyan-300',
        pink: 'border-pink-300',
        indigo: 'border-indigo-300',
        amber: 'border-amber-300',
        red: 'border-red-300',
        emerald: 'border-emerald-300',
      }
      return borderColors[color] || 'border-gray-300'
    }
    if (status === 'error') return 'border-red-300'
    if (status === 'pending') return 'border-yellow-300'
    return 'border-gray-200'
  }

  const successCount = pipelines.filter(p => results[p]?.status === 'success').length
  const agreementPercent = (successCount / pipelines.length * 100).toFixed(0)

  return (
    <div className="space-y-6">
      {/* Pipeline Grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Pipeline Results</h3>
          <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
            Click to explore →
          </span>
        </div>
        <div className="grid grid-cols-5 gap-3">
          {pipelines.map((key) => {
            const config = PIPELINE_CONFIG[key as PipelineKey]
            const result = results[key]
            const metric = getMetricDisplay(key, result?.data)
            const Icon = config.icon

            return (
              <button
                key={key}
                onClick={() => onTabChange(key as PipelineKey)}
                className={`p-3 rounded-lg border-2 text-left transition-all duration-200 hover:shadow-lg hover:scale-[1.03] active:scale-[0.98] group ${
                  result?.status === 'success'
                    ? `${getColorClass(config.color, 'bg')} ${getStatusBorderColor(result.status, config.color)} hover:shadow-${config.color}-200`
                    : result?.status === 'error'
                    ? 'bg-red-50 border-red-300'
                    : result?.status === 'pending'
                    ? 'bg-yellow-50 border-yellow-300'
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className={`p-1.5 rounded-lg ${getColorClass(config.color, 'bg')} group-hover:scale-110 transition-transform`}>
                    <Icon className={`w-4 h-4 ${getColorClass(config.color, 'text')}`} />
                  </div>
                  {result && <StatusIcon status={result.status} />}
                </div>
                <div className="text-sm font-semibold text-gray-800">{config.name}</div>
                {result?.status === 'success' && (
                  <div className="mt-2">
                    <span className="text-xl font-bold text-gray-900">{metric.value}</span>
                    <span className="text-xs text-gray-500 ml-1">{metric.label}</span>
                  </div>
                )}
                {result?.status === 'error' && (
                  <div className="mt-2 text-xs text-red-600 font-medium">Error</div>
                )}
                {result?.status === 'pending' && (
                  <div className="mt-2 text-xs text-yellow-600 font-medium">Processing...</div>
                )}
                {!result && (
                  <div className="mt-2 text-xs text-gray-400">Not available</div>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Model Agreement Progress */}
      <div className="bg-gray-50 rounded-lg p-4 border">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-gray-700">Processing Progress</h3>
          <span className="text-sm font-medium text-gray-600">{successCount}/{pipelines.length} completed</span>
        </div>
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-400 to-green-500 transition-all duration-500 rounded-full"
            style={{ width: `${agreementPercent}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-500">
          <span>0%</span>
          <span className="font-medium text-green-600">{agreementPercent}% complete</span>
          <span>100%</span>
        </div>
      </div>

      {/* Recommendation */}
      {results.fusion?.data?.fusion_result?.recommendation && (
        <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Info className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <h4 className="font-semibold text-blue-800 mb-1">Recommendation</h4>
              <p className="text-blue-700 text-sm">{results.fusion.data.fusion_result.recommendation}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Status Icon component
function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'success':
      return <CheckCircle className="w-4 h-4 text-green-500" />
    case 'error':
      return <XCircle className="w-4 h-4 text-red-500" />
    case 'pending':
      return <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />
    default:
      return <AlertCircle className="w-4 h-4 text-gray-400" />
  }
}

// Get color classes helper
function getColorClass(color: string, type: 'bg' | 'border' | 'text') {
  const colors: Record<string, Record<string, string>> = {
    blue: { bg: 'bg-blue-100', border: 'border-blue-500', text: 'text-blue-600' },
    yellow: { bg: 'bg-yellow-100', border: 'border-yellow-500', text: 'text-yellow-600' },
    purple: { bg: 'bg-purple-100', border: 'border-purple-500', text: 'text-purple-600' },
    green: { bg: 'bg-green-100', border: 'border-green-500', text: 'text-green-600' },
    orange: { bg: 'bg-orange-100', border: 'border-orange-500', text: 'text-orange-600' },
    cyan: { bg: 'bg-cyan-100', border: 'border-cyan-500', text: 'text-cyan-600' },
    pink: { bg: 'bg-pink-100', border: 'border-pink-500', text: 'text-pink-600' },
    indigo: { bg: 'bg-indigo-100', border: 'border-indigo-500', text: 'text-indigo-600' },
    amber: { bg: 'bg-amber-100', border: 'border-amber-500', text: 'text-amber-600' },
    red: { bg: 'bg-red-100', border: 'border-red-500', text: 'text-red-600' },
    emerald: { bg: 'bg-emerald-100', border: 'border-emerald-500', text: 'text-emerald-600' },
    violet: { bg: 'bg-violet-100', border: 'border-violet-500', text: 'text-violet-600' },
  }
  return colors[color]?.[type] || 'bg-gray-100'
}

// YOLO Tab
function YoloTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="YOLO" />

  const features = data.features || {}

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Object Detection Results</h3>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Total Detections" value={features.num_detections || 0} />
        <MetricCard label="Avg Confidence" value={`${((features.avg_confidence || 0) * 100).toFixed(1)}%`} />
        <MetricCard label="Max Confidence" value={`${((features.max_confidence || 0) * 100).toFixed(1)}%`} />
        <MetricCard label="Detection Rate" value={`${((features.detection_rate || 0) * 100).toFixed(1)}%`} />
      </div>

      {/* Position Stats */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Avg Box Width" value={Math.round(features.avg_box_width || 0)} unit="px" />
        <MetricCard label="Avg Box Height" value={Math.round(features.avg_box_height || 0)} unit="px" />
        <MetricCard label="Position Stability" value={((features.position_stability || 0) * 100).toFixed(1)} unit="%" />
      </div>

      {/* Detection samples */}
      {data.detections && data.detections.length > 0 && (
        <div>
          <h4 className="font-medium mb-2">Sample Detections</h4>
          <div className="max-h-48 overflow-y-auto border rounded">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left">Frame</th>
                  <th className="px-3 py-2 text-left">Count</th>
                  <th className="px-3 py-2 text-left">Avg Conf</th>
                </tr>
              </thead>
              <tbody>
                {data.detections.slice(0, 10).map((det: any, i: number) => (
                  <tr key={i} className="border-t">
                    <td className="px-3 py-2">{det.frame}</td>
                    <td className="px-3 py-2">{det.detections?.length || 0}</td>
                    <td className="px-3 py-2">
                      {det.detections?.length > 0
                        ? `${(det.detections.reduce((sum: number, d: any) => sum + (d.confidence || 0), 0) / det.detections.length * 100).toFixed(1)}%`
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// SAM3 Tab
function Sam3Tab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="SAM3" />

  const features = data.features || {}

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Segmentation Results</h3>

      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Avg Area Ratio" value={((features.avg_area_ratio || 0) * 100).toFixed(1)} unit="%" />
        <MetricCard label="Avg Circularity" value={(features.avg_circularity || 0).toFixed(3)} />
        <MetricCard label="Avg Aspect Ratio" value={(features.avg_aspect_ratio || 0).toFixed(2)} />
      </div>

      {data.segmentations && (
        <div>
          <h4 className="font-medium mb-2">Segmentation Coverage</h4>
          <p className="text-sm text-gray-600">
            {data.segmentations.filter((s: any) => s.mask_available).length} of {data.segmentations.length} frames have masks
          </p>
        </div>
      )}
    </div>
  )
}

// DINOv3 Tab
function DinOv3Tab({ data, videoId }: { data: any; videoId: string }) {
  if (!data) return <NotAvailable pipeline="DINOv3" />

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Visual Embeddings</h3>

      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Embedding Dim" value={data.embedding_dim || 768} />
        <MetricCard label="Num Embeddings" value={data.num_embeddings || 0} />
        <MetricCard label="Neighbor Evidence" value={((data.neighbor_evidence || 0) * 100).toFixed(1)} unit="%" />
      </div>

      {/* Similar Videos */}
      {data.similar_cases && data.similar_cases.length > 0 && (
        <div>
          <h4 className="font-medium mb-3">Similar Videos</h4>
          <div className="space-y-2">
            {data.similar_cases.slice(0, 5).map((sim: any, i: number) => (
              <div key={i} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <span className="font-mono text-sm">{sim.video_id?.substring(0, 8)}...</span>
                <span className="text-sm">
                  Similarity: <span className="font-medium">{((sim.score || 0) * 100).toFixed(1)}%</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// T-LEAP Tab
function TleapTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="T-LEAP" />

  const loco = data.locomotion_features || {}

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Pose Estimation & Locomotion</h3>

      {/* Key Clinical Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Lameness Score"
          value={(loco.lameness_score || 0).toFixed(2)}
          highlight={loco.lameness_score > 0.5}
        />
        <MetricCard label="Head Bob" value={(loco.head_bob_magnitude || 0).toFixed(3)} />
        <MetricCard label="Back Arch" value={(loco.back_arch_mean || 0).toFixed(3)} />
        <MetricCard label="Steadiness" value={(loco.steadiness_score || 0).toFixed(2)} />
      </div>

      {/* Asymmetry */}
      <div className="grid grid-cols-2 gap-4">
        <MetricCard label="Front Leg Asymmetry" value={(loco.front_leg_asymmetry || 0).toFixed(3)} />
        <MetricCard label="Rear Leg Asymmetry" value={(loco.rear_leg_asymmetry || 0).toFixed(3)} />
      </div>

      {/* Processing Info */}
      <div className="text-sm text-gray-600">
        <p>Frames processed: {data.frames_processed || 0} / {data.total_frames || 0}</p>
        <p>FPS: {data.fps || 0}</p>
      </div>
    </div>
  )
}

// TCN Tab
function TcnTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="TCN" />

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Temporal Convolutional Network</h3>

      {/* Severity Gauge */}
      <div className="flex items-center justify-center">
        <div className="relative w-48 h-48">
          <svg viewBox="0 0 100 100" className="transform -rotate-90">
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="#e5e7eb"
              strokeWidth="8"
            />
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke={data.severity_score > 0.5 ? '#ef4444' : '#22c55e'}
              strokeWidth="8"
              strokeDasharray={`${(data.severity_score || 0) * 251.2} 251.2`}
              strokeLinecap="round"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center flex-col">
            <span className="text-3xl font-bold">{((data.severity_score || 0) * 100).toFixed(0)}%</span>
            <span className="text-sm text-gray-500">Severity</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Uncertainty" value={((data.uncertainty || 0) * 100).toFixed(1)} unit="%" />
        <MetricCard label="Input Frames" value={data.input_frames || 0} />
        <MetricCard label="Confidence" value={((data.confidence || 0) * 100).toFixed(1)} unit="%" />
      </div>
    </div>
  )
}

// Transformer Tab
function TransformerTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="Transformer" />

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Gait Transformer</h3>

      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          label="Severity Score"
          value={(data.severity_score || 0).toFixed(3)}
          highlight={data.severity_score > 0.5}
        />
        <MetricCard label="Uncertainty" value={((data.uncertainty || 0) * 100).toFixed(1)} unit="%" />
        <MetricCard label="Prediction" value={data.prediction === 1 ? 'Lame' : 'Healthy'} />
      </div>

      {data.temporal_saliency && data.temporal_saliency.length > 0 && (
        <div>
          <h4 className="font-medium mb-2">Temporal Saliency</h4>
          <p className="text-sm text-gray-600">Shows which frames contributed most to the prediction</p>
        </div>
      )}
    </div>
  )
}

// GNN Tab
function GnnTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="GNN" />

  const graphInfo = data.graph_info || {}

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Graph Neural Network (GraphGPS)</h3>

      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Severity Score" value={(data.severity_score || 0).toFixed(3)} highlight={data.severity_score > 0.5} />
        <MetricCard label="Uncertainty" value={((data.uncertainty || 0) * 100).toFixed(1)} unit="%" />
        <MetricCard label="Graph Nodes" value={graphInfo.num_nodes || 0} />
        <MetricCard label="Graph Edges" value={graphInfo.num_edges || 0} />
      </div>

      {/* Neighbor Influence */}
      {data.neighbor_influence && data.neighbor_influence.length > 0 && (
        <div>
          <h4 className="font-medium mb-3">Neighbor Influence</h4>
          <div className="space-y-2">
            {data.neighbor_influence.slice(0, 5).map((neighbor: any, i: number) => (
              <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="font-mono text-sm">{neighbor.video_id?.substring(0, 8)}...</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-gray-200 rounded overflow-hidden">
                    <div
                      className="h-full bg-indigo-500"
                      style={{ width: `${(neighbor.score || 0) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm w-12 text-right">{((neighbor.score || 0) * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Graph Transformer Tab (NEW - for Graphormer)
function GraphTransformerTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="Graph Transformer" />

  const graphInfo = data.graph_info || {}
  const attentionInfo = data.attention_info || {}

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Graph Transformer (Graphormer)</h3>

      {/* Model Info Banner */}
      <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm">
        <span className="font-medium text-amber-800">Model:</span>{' '}
        <span className="text-amber-700">{data.model || 'CowLamenessGraphormer'}</span>
      </div>

      {/* Key Predictions */}
      <div className="grid grid-cols-2 gap-4">
        <div className={`p-4 rounded-lg ${data.graph_prediction > 0.5 ? 'bg-red-50' : 'bg-green-50'}`}>
          <div className="text-sm text-gray-600 mb-1">Graph-Level Prediction</div>
          <div className={`text-2xl font-bold ${data.graph_prediction > 0.5 ? 'text-red-600' : 'text-green-600'}`}>
            {((data.graph_prediction || 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className={`p-4 rounded-lg ${data.node_prediction > 0.5 ? 'bg-red-50' : 'bg-green-50'}`}>
          <div className="text-sm text-gray-600 mb-1">Node-Level Prediction</div>
          <div className={`text-2xl font-bold ${data.node_prediction > 0.5 ? 'text-red-600' : 'text-green-600'}`}>
            {((data.node_prediction || 0) * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Uncertainty" value={((data.uncertainty || 0) * 100).toFixed(1)} unit="%" />
        <MetricCard label="Confidence" value={((data.confidence || 0) * 100).toFixed(1)} unit="%" />
        <MetricCard label="Nodes" value={graphInfo.num_nodes || 0} />
        <MetricCard label="Edges" value={graphInfo.num_edges || 0} />
      </div>

      {/* Architecture Info */}
      <div>
        <h4 className="font-medium mb-2">Architecture</h4>
        <div className="grid grid-cols-3 gap-3 text-sm">
          <div className="p-2 bg-gray-50 rounded">
            <span className="text-gray-500">Layers:</span> <span className="font-medium">{graphInfo.num_layers || 6}</span>
          </div>
          <div className="p-2 bg-gray-50 rounded">
            <span className="text-gray-500">Heads:</span> <span className="font-medium">{graphInfo.num_heads || 8}</span>
          </div>
          <div className="p-2 bg-gray-50 rounded">
            <span className="text-gray-500">Hidden Dim:</span> <span className="font-medium">{graphInfo.hidden_dim || 128}</span>
          </div>
        </div>
      </div>

      {/* Top Attending Nodes */}
      {attentionInfo.top_attending_nodes && attentionInfo.top_attending_nodes.length > 0 && (
        <div>
          <h4 className="font-medium mb-3">Top Attending Nodes</h4>
          <div className="space-y-2">
            {attentionInfo.top_attending_nodes.map((node: any, i: number) => (
              <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="font-mono text-sm">{node.video_id?.substring(0, 8)}...</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-2 bg-gray-200 rounded overflow-hidden">
                    <div
                      className="h-full bg-amber-500"
                      style={{ width: `${(node.attention || 0) * 100 * 10}%` }} // Scale up for visibility
                    />
                  </div>
                  <span className="text-sm w-16 text-right">{(node.attention || 0).toFixed(4)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ML Ensemble Tab
function MlEnsembleTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="ML Ensemble" />

  const predictions = data.predictions || {}
  const ensemble = predictions.ensemble || {}

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">ML Ensemble (CatBoost, XGBoost, LightGBM)</h3>

      {/* Ensemble Result */}
      <div className={`p-4 rounded-lg ${ensemble.prediction === 1 ? 'bg-red-50' : 'bg-green-50'}`}>
        <div className="text-sm text-gray-600 mb-1">Ensemble Prediction</div>
        <div className="flex items-center justify-between">
          <div className={`text-2xl font-bold ${ensemble.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
            {ensemble.prediction === 1 ? 'LAME' : 'HEALTHY'}
          </div>
          <div className="text-xl font-semibold">
            {((ensemble.probability || 0) * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Individual Model Predictions */}
      <div>
        <h4 className="font-medium mb-3">Individual Model Predictions</h4>
        <div className="grid grid-cols-3 gap-4">
          {['catboost', 'xgboost', 'lightgbm'].map((model) => (
            <div key={model} className="p-3 border rounded-lg">
              <div className="text-sm font-medium capitalize mb-2">{model}</div>
              {predictions[model] ? (
                <>
                  <div className="text-lg font-semibold">
                    {((predictions[model].probability || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    Weight: {(ensemble.weights?.[model] || 0.33).toFixed(2)}
                  </div>
                </>
              ) : (
                <div className="text-gray-400">N/A</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Features Used */}
      {data.feature_names && data.features && (
        <div>
          <h4 className="font-medium mb-2">Input Features ({data.feature_names.length})</h4>
          <div className="max-h-48 overflow-y-auto border rounded text-sm">
            <table className="w-full">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left">Feature</th>
                  <th className="px-3 py-2 text-right">Value</th>
                </tr>
              </thead>
              <tbody>
                {data.feature_names.slice(0, 20).map((name: string, i: number) => (
                  <tr key={i} className="border-t">
                    <td className="px-3 py-1">{name}</td>
                    <td className="px-3 py-1 text-right font-mono">
                      {typeof data.features[i] === 'number'
                        ? data.features[i].toFixed(4)
                        : data.features[i]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// Fusion Tab
function FusionTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="Fusion" />

  const result = data.fusion_result || {}
  const contributions = result.pipeline_contributions || {}

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Fusion Service (Final Decision)</h3>

      {/* Final Prediction */}
      <div className={`p-6 rounded-lg text-center ${result.final_prediction === 1 ? 'bg-red-50' : 'bg-green-50'}`}>
        <div className={`text-4xl font-bold mb-2 ${result.final_prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
          {result.final_prediction === 1 ? 'LAME' : 'HEALTHY'}
        </div>
        <div className="text-xl">
          {((result.final_probability || 0) * 100).toFixed(1)}% probability
        </div>
        <div className="text-sm text-gray-600 mt-2">
          Confidence: {((result.confidence || 0) * 100).toFixed(0)}%
        </div>
      </div>

      {/* Decision Info */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-3 bg-gray-50 rounded">
          <div className="text-sm text-gray-500">Decision Mode</div>
          <div className="font-medium">{result.decision_mode || 'hybrid'}</div>
        </div>
        <div className="p-3 bg-gray-50 rounded">
          <div className="text-sm text-gray-500">Model Agreement</div>
          <div className="font-medium">{((result.model_agreement || 0) * 100).toFixed(0)}%</div>
        </div>
      </div>

      {/* Pipeline Contributions */}
      <div>
        <h4 className="font-medium mb-3">Pipeline Contributions</h4>
        <div className="space-y-3">
          {Object.entries(contributions).map(([pipeline, contrib]: [string, any]) => (
            <div key={pipeline} className="flex items-center gap-3">
              <div className="w-24 text-sm font-medium">{pipeline}</div>
              <div className="flex-1 h-4 bg-gray-200 rounded overflow-hidden">
                <div
                  className={`h-full ${contrib.prediction === 1 ? 'bg-red-400' : 'bg-green-400'}`}
                  style={{ width: `${(contrib.probability || 0) * 100}%` }}
                />
              </div>
              <div className="w-16 text-sm text-right">
                {((contrib.probability || 0) * 100).toFixed(0)}%
              </div>
              <div className="w-12 text-xs text-gray-500">
                w:{contrib.weight?.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendation */}
      {result.recommendation && (
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-medium text-blue-800 mb-1">Recommendation</h4>
          <p className="text-blue-700">{result.recommendation}</p>
        </div>
      )}
    </div>
  )
}

// SHAP Tab
function ShapTab({ data }: { data: any }) {
  if (!data) return <NotAvailable pipeline="SHAP" />

  const shapValues = data.shap_values || []

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">SHAP Explainability</h3>

      <div className="grid grid-cols-2 gap-4">
        <MetricCard label="Base Value" value={(data.base_value || 0).toFixed(4)} />
        <MetricCard label="Prediction" value={(data.prediction || 0).toFixed(4)} />
      </div>

      {/* Top Features */}
      {shapValues.length > 0 && (
        <div>
          <h4 className="font-medium mb-3">Top Feature Contributions</h4>
          <div className="space-y-2">
            {shapValues
              .sort((a: any, b: any) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
              .slice(0, 10)
              .map((item: any, i: number) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-32 text-sm truncate" title={item.feature}>
                    {item.feature}
                  </div>
                  <div className="flex-1 flex items-center">
                    <div className="w-1/2 flex justify-end">
                      {item.shap_value < 0 && (
                        <div
                          className="h-4 bg-green-400 rounded-l"
                          style={{ width: `${Math.min(100, Math.abs(item.shap_value) * 1000)}%` }}
                        />
                      )}
                    </div>
                    <div className="w-px h-6 bg-gray-300" />
                    <div className="w-1/2">
                      {item.shap_value > 0 && (
                        <div
                          className="h-4 bg-red-400 rounded-r"
                          style={{ width: `${Math.min(100, Math.abs(item.shap_value) * 1000)}%` }}
                        />
                      )}
                    </div>
                  </div>
                  <div className="w-20 text-xs text-right font-mono">
                    {item.shap_value?.toFixed(4)}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Helper Components
function MetricCard({ label, value, unit, highlight }: { label: string; value: string | number; unit?: string; highlight?: boolean }) {
  return (
    <div className={`p-3 rounded-lg ${highlight ? 'bg-red-50' : 'bg-gray-50'}`}>
      <div className="text-sm text-gray-500">{label}</div>
      <div className={`text-lg font-semibold ${highlight ? 'text-red-600' : ''}`}>
        {value}{unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}
      </div>
    </div>
  )
}

function NotAvailable({ pipeline }: { pipeline: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 text-gray-500">
      <AlertCircle className="w-12 h-12 mb-4" />
      <p>{pipeline} results not available</p>
      <p className="text-sm">This pipeline may not have processed this video yet.</p>
    </div>
  )
}
