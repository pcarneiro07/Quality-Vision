import { useState, useEffect, useRef } from 'react'
import type { EpochLog } from '../types'
import { fetchTrainingMetrics } from './api'

const POLL_INTERVAL_MS = 2000

export function useTrainingLogs() {
  const [logs, setLogs] = useState<EpochLog[]>([])
  const [isPolling, setIsPolling] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchLogs = async () => {
    try {
      const data = await fetchTrainingMetrics()
      setLogs(data.epochs ?? [])
      setError(null)
    } catch (e) {
      setError('Não foi possível conectar à API. Verifique se o backend está rodando.')
    }
  }

  useEffect(() => {
    fetchLogs()
    if (isPolling) {
      intervalRef.current = setInterval(fetchLogs, POLL_INTERVAL_MS)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [isPolling])

  return { logs, isPolling, setIsPolling, error, refetch: fetchLogs }
}
