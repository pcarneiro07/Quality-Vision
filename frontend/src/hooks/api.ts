import type { TrainingMetrics, PredictionResult, EvaluationResults } from '../types'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Erro desconhecido' }))
    throw new Error(err.detail ?? 'Falha na predição')
  }
  return res.json()
}

export async function predictImageGradCAM(file: File): Promise<PredictionResult & { gradcam_base64: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_URL}/predict-gradcam`, { method: 'POST', body: formData })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Erro desconhecido' }))
    throw new Error(err.detail ?? 'Falha na predição')
  }
  return res.json()
}

export async function fetchTrainingMetrics(): Promise<TrainingMetrics> {
  const res = await fetch(`${API_URL}/metrics`)
  if (!res.ok) throw new Error('Falha ao buscar métricas')
  return res.json()
}

export async function fetchEvaluationResults(): Promise<EvaluationResults> {
  const res = await fetch(`${API_URL}/confusion-matrix`)
  if (!res.ok) throw new Error('Resultados de avaliação não encontrados')
  return res.json()
}

export async function fetchHealth(): Promise<{ status: string; model_loaded: boolean; device: string }> {
  const res = await fetch(`${API_URL}/health`)
  if (!res.ok) throw new Error('API indisponível')
  return res.json()
}