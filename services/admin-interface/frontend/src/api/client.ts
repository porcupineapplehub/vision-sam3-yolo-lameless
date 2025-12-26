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
  getGlobal: async () => {
    const response = await apiClient.get('/api/shap/global')
    return response.data
  },
}

