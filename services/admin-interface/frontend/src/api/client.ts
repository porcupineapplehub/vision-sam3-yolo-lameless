import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Video endpoints
export const videosApi = {
  upload: async (file: File, label?: number, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    formData.append('file', file)
    if (label !== undefined && label !== null) {
      formData.append('label', String(label))
    }
    const response = await apiClient.post('/api/videos/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      },
    })
    return response.data
  },
  uploadMultiple: async (
    files: Array<{ file: File; label?: number }>,
    onFileProgress?: (index: number, progress: number) => void,
    onFileComplete?: (index: number, result: any) => void
  ) => {
    const results = []
    for (let i = 0; i < files.length; i++) {
      const { file, label } = files[i]
      try {
        const result = await videosApi.upload(file, label, (progress) => {
          onFileProgress?.(i, progress)
        })
        onFileComplete?.(i, { success: true, data: result })
        results.push({ success: true, data: result })
      } catch (error: any) {
        onFileComplete?.(i, { success: false, error: error.response?.data?.detail || 'Upload failed' })
        results.push({ success: false, error: error.response?.data?.detail || 'Upload failed' })
      }
    }
    return results
  },
  get: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}`)
    return response.data
  },
  list: async (skip = 0, limit = 100) => {
    const response = await apiClient.get('/api/videos', { params: { skip, limit } })
    return response.data
  },
  getStreamUrl: (videoId: string) => {
    return `${API_BASE_URL}/api/videos/${videoId}/stream`
  },
  getAnnotatedUrl: (videoId: string) => {
    return `${API_BASE_URL}/api/videos/${videoId}/annotated`
  },
  getFrameUrl: (videoId: string, frameNum: number, annotated = false) => {
    return `${API_BASE_URL}/api/videos/${videoId}/frame/${frameNum}?annotated=${annotated}`
  },
  triggerAnnotation: async (videoId: string, options?: {
    include_yolo?: boolean
    include_pose?: boolean
    show_confidence?: boolean
    show_labels?: boolean
  }) => {
    const response = await apiClient.post(`/api/videos/${videoId}/annotate`, null, {
      params: options || {}
    })
    return response.data
  },
  getAnnotationStatus: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}/annotation-status`)
    return response.data
  },
  deleteAnnotation: async (videoId: string) => {
    const response = await apiClient.delete(`/api/videos/${videoId}/annotation`)
    return response.data
  },
  getDetections: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}/detections`)
    return response.data
  },
  getPose: async (videoId: string) => {
    const response = await apiClient.get(`/api/videos/${videoId}/pose`)
    return response.data
  },
}

// Analysis endpoints
export const analysisApi = {
  get: async (videoId: string) => {
    const response = await apiClient.get(`/api/analysis/${videoId}`)
    return response.data
  },
  getSummary: async (videoId: string) => {
    const response = await apiClient.get(`/api/analysis/${videoId}/summary`)
    return response.data
  },
}

// Training endpoints
export const trainingApi = {
  label: async (videoId: string, label: number, confidence = 'certain') => {
    const response = await apiClient.post(`/api/training/videos/${videoId}/label`, {
      label,
      confidence,
    })
    return response.data
  },
  getQueue: async () => {
    const response = await apiClient.get('/api/training/queue')
    return response.data
  },
  getStats: async () => {
    const response = await apiClient.get('/api/training/stats')
    return response.data
  },
  getStatus: async () => {
    const response = await apiClient.get('/api/training/status')
    return response.data
  },
  getModels: async () => {
    const response = await apiClient.get('/api/training/models')
    return response.data
  },
  startMLTraining: async () => {
    const response = await apiClient.post('/api/training/ml/start')
    return response.data
  },
  startYOLOTraining: async () => {
    const response = await apiClient.post('/api/training/yolo/start')
    return response.data
  },
  // Pairwise comparison endpoints
  submitPairwise: async (videoId1: string, videoId2: string, winner: number, confidence = 'confident') => {
    const response = await apiClient.post('/api/training/pairwise', {
      video_id_1: videoId1,
      video_id_2: videoId2,
      winner,
      confidence,
    })
    return response.data
  },
  getNextPairwise: async (excludeCompleted = true) => {
    const response = await apiClient.get('/api/training/pairwise/next', {
      params: { exclude_completed: excludeCompleted }
    })
    return response.data
  },
  getPairwiseStats: async () => {
    const response = await apiClient.get('/api/training/pairwise/stats')
    return response.data
  },
  getPairwiseRanking: async () => {
    const response = await apiClient.get('/api/training/pairwise/ranking')
    return response.data
  },
}

// Model endpoints
export const modelsApi = {
  getParameters: async () => {
    const response = await apiClient.get('/api/models/parameters')
    return response.data
  },
  updateParameters: async (parameters: any) => {
    const response = await apiClient.post('/api/models/parameters', parameters)
    return response.data
  },
  getComparison: async () => {
    const response = await apiClient.get('/api/models/comparison')
    return response.data
  },
}

// SHAP endpoints
export const shapApi = {
  getLocal: async (videoId: string) => {
    const response = await apiClient.get(`/api/shap/${videoId}/local`)
    return response.data
  },
  getForcePlot: async (videoId: string) => {
    const response = await apiClient.get(`/api/shap/${videoId}/force-plot`)
    return response.data
  },
  getGlobal: async () => {
    const response = await apiClient.get('/api/shap/global')
    return response.data
  },
}
