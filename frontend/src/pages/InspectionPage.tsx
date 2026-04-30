import { useState, useCallback, useRef } from 'react'
import { Upload, CheckCircle, XCircle, Loader, AlertTriangle } from 'lucide-react'
import type { PredictionResult } from '../types'
import { predictImageGradCAM } from '../hooks/api'

type ResultWithGradCAM = PredictionResult & { gradcam_base64: string }

export function InspectionPage() {
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<ResultWithGradCAM | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showGradCAM, setShowGradCAM] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Arquivo deve ser uma imagem (JPG, PNG, BMP)')
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target?.result as string)
    reader.readAsDataURL(file)

    setLoading(true)
    setResult(null)
    setError(null)
    setShowGradCAM(false)

    try {
      const res = await predictImageGradCAM(file)
      setResult(res)
    } catch (e: any) {
      setError(e.message ?? 'Erro ao processar imagem')
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const reset = () => {
    setResult(null)
    setPreview(null)
    setError(null)
    setShowGradCAM(false)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div style={styles.container}>
      <div
        style={{
          ...styles.dropZone,
          ...(dragOver ? styles.dropZoneActive : {}),
          ...(result ? styles.dropZoneSmall : {}),
        }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !loading && inputRef.current?.click()}
      >
        <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={onInputChange} />

        {preview ? (
          <div style={styles.previewWrapper}>
            <img src={preview} alt="preview" style={styles.previewImg} />
            {loading && (
              <div style={styles.previewOverlay}>
                <Loader size={32} style={styles.spinner} />
                <span style={styles.loadingText}>Analisando peça...</span>
              </div>
            )}
          </div>
        ) : (
          <div style={styles.uploadContent}>
            <Upload size={40} color="var(--accent)" strokeWidth={1.5} />
            <p style={styles.uploadTitle}>Arraste uma imagem aqui</p>
            <p style={styles.uploadSub}>ou clique para selecionar</p>
            <p style={styles.uploadHint}>JPG, PNG, BMP — max 10MB</p>
          </div>
        )}
      </div>

      {result && !loading && (
        <div style={{
          ...styles.resultCard,
          ...(result.result === 'APROVADA' ? styles.resultOk : styles.resultDefect),
        }}>
          <div style={styles.resultHeader}>
            {result.result === 'APROVADA'
              ? <CheckCircle size={52} color="var(--accent)" strokeWidth={1.5} />
              : <XCircle size={52} color="var(--danger)" strokeWidth={1.5} />
            }
            <div>
              <p style={styles.resultLabel}>RESULTADO DA INSPEÇÃO</p>
              <h2 style={{ ...styles.resultTitle, color: result.result === 'APROVADA' ? 'var(--accent)' : 'var(--danger)' }}>
                {result.result}
              </h2>
            </div>
          </div>

          <div style={styles.metricsRow}>
            <div style={styles.metricBox}>
              <span style={styles.metricLabel}>Confiança</span>
              <span style={styles.metricValue}>{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <div style={styles.metricBox}>
              <span style={styles.metricLabel}>P(Defeito)</span>
              <span style={{ ...styles.metricValue, color: 'var(--danger)' }}>{(result.probability_defect * 100).toFixed(1)}%</span>
            </div>
            <div style={styles.metricBox}>
              <span style={styles.metricLabel}>P(OK)</span>
              <span style={{ ...styles.metricValue, color: 'var(--accent)' }}>{(result.probability_ok * 100).toFixed(1)}%</span>
            </div>
          </div>

          <div style={styles.probBarWrapper}>
            <div style={styles.probBarLabels}>
              <span style={{ color: 'var(--accent)', fontSize: 11 }}>✓ OK</span>
              <span style={{ color: 'var(--danger)', fontSize: 11 }}>✗ DEFEITO</span>
            </div>
            <div style={styles.probBarTrack}>
              <div style={{ ...styles.probBarFill, width: `${result.probability_ok * 100}%` }} />
            </div>
          </div>

          <div style={styles.gradcamSection}>
            <button style={styles.gradcamToggle} onClick={() => setShowGradCAM(v => !v)}>
              {showGradCAM ? '▲ Ocultar' : '▼ Ver'} mapa de atenção
            </button>

            {showGradCAM && (
              <div style={styles.gradcamWrapper}>
                <div style={styles.gradcamCard}>
                  <p style={styles.gradcamLabel}>Imagem original</p>
                  <img src={preview!} alt="original" style={styles.gradcamImg} />
                </div>
                <div style={styles.gradcamCard}>
                  <p style={styles.gradcamLabel}>Regiões analisadas</p>
                  <img src={`data:image/jpeg;base64,${result.gradcam_base64}`} alt="mapa de atenção" style={styles.gradcamImg} />
                </div>
                <div style={styles.gradcamLegend}>
                  <div style={styles.legendBar} />
                  <div style={styles.legendLabels}>
                    <span>Baixa atenção</span>
                    <span>Alta atenção</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <button style={styles.resetBtn} onClick={reset}>
            Inspecionar Nova Peça
          </button>
        </div>
      )}

      {error && (
        <div style={styles.errorCard}>
          <AlertTriangle size={20} color="var(--warning)" />
          <span>{error}</span>
        </div>
      )}

      {!result && !loading && (
        <div style={styles.tips}>
          <p style={styles.tipsTitle}>💡 Dicas para teste</p>
          <p style={styles.tipText}>Use imagens da pasta <code style={styles.code}>data/raw/casting_data/test/</code></p>
          <p style={styles.tipText}>
            <span style={{ color: 'var(--accent)' }}>ok_front/</span> → peças sem defeito &nbsp;|&nbsp;
            <span style={{ color: 'var(--danger)' }}>def_front/</span> → peças com defeito
          </p>
        </div>
      )}
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 720, margin: '0 auto', padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' },
  dropZone: { border: '2px dashed var(--border-light)', borderRadius: 16, padding: '3rem 2rem', cursor: 'pointer', transition: 'all 0.2s', background: 'var(--surface)', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 260, position: 'relative', overflow: 'hidden' },
  dropZoneActive: { borderColor: 'var(--accent)', background: 'var(--accent-dim)' },
  dropZoneSmall: { minHeight: 200, padding: '1.5rem' },
  uploadContent: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' },
  uploadTitle: { fontSize: 18, fontWeight: 600, color: 'var(--text)' },
  uploadSub: { fontSize: 14, color: 'var(--text-muted)' },
  uploadHint: { fontSize: 12, color: 'var(--text-muted)', marginTop: '0.5rem', fontFamily: 'var(--font-mono)' },
  previewWrapper: { position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' },
  previewImg: { maxHeight: 180, maxWidth: '100%', borderRadius: 8, objectFit: 'contain', filter: 'brightness(0.9)' },
  previewOverlay: { position: 'absolute', inset: 0, background: 'rgba(10,12,16,0.7)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', borderRadius: 8 },
  spinner: { animation: 'spin 1s linear infinite', color: 'var(--accent)' },
  loadingText: { color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontSize: 13 },
  resultCard: { borderRadius: 16, padding: '1.75rem', border: '1px solid', display: 'flex', flexDirection: 'column', gap: '1.25rem' },
  resultOk: { background: 'rgba(0, 212, 170, 0.06)', borderColor: 'rgba(0, 212, 170, 0.3)' },
  resultDefect: { background: 'rgba(255, 71, 87, 0.06)', borderColor: 'rgba(255, 71, 87, 0.3)' },
  resultHeader: { display: 'flex', alignItems: 'center', gap: '1.25rem' },
  resultLabel: { fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.1em', marginBottom: 4 },
  resultTitle: { fontSize: 36, fontWeight: 700, fontFamily: 'var(--font-mono)', letterSpacing: '-0.02em' },
  metricsRow: { display: 'flex', gap: '1rem' },
  metricBox: { flex: 1, background: 'var(--surface-2)', borderRadius: 10, padding: '0.875rem', display: 'flex', flexDirection: 'column', gap: 4, border: '1px solid var(--border)' },
  metricLabel: { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase' },
  metricValue: { fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text)' },
  probBarWrapper: { display: 'flex', flexDirection: 'column', gap: 6 },
  probBarLabels: { display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)' },
  probBarTrack: { height: 8, borderRadius: 4, background: 'var(--danger-dim)', overflow: 'hidden', border: '1px solid var(--border)' },
  probBarFill: { height: '100%', background: 'var(--accent)', borderRadius: 4, transition: 'width 0.6s ease' },
  gradcamSection: { display: 'flex', flexDirection: 'column', gap: '0.75rem' },
  gradcamToggle: { background: 'transparent', border: '1px solid var(--border-light)', borderRadius: 8, color: 'var(--text-muted)', padding: '0.5rem 1rem', cursor: 'pointer', fontSize: 13, fontFamily: 'var(--font-mono)', alignSelf: 'flex-start', transition: 'all 0.2s' },
  gradcamWrapper: { display: 'flex', flexDirection: 'column', gap: '0.75rem' },
  gradcamCard: { display: 'flex', flexDirection: 'column', gap: '0.5rem', background: 'var(--surface-2)', borderRadius: 10, padding: '1rem', border: '1px solid var(--border)', alignItems: 'center' },
  gradcamLabel: { fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', alignSelf: 'flex-start' },
  gradcamImg: { width: '100%', maxWidth: 400, borderRadius: 8, objectFit: 'contain' },
  gradcamLegend: { display: 'flex', flexDirection: 'column', gap: 4, padding: '0 0.25rem' },
  legendBar: { height: 8, borderRadius: 4, background: 'linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000)' },
  legendLabels: { display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' },
  resetBtn: { background: 'var(--surface-2)', border: '1px solid var(--border-light)', borderRadius: 8, color: 'var(--text)', padding: '0.75rem 1.5rem', cursor: 'pointer', fontSize: 14, fontWeight: 500, transition: 'all 0.2s', alignSelf: 'flex-start' },
  errorCard: { background: 'rgba(255, 165, 2, 0.08)', border: '1px solid rgba(255, 165, 2, 0.3)', borderRadius: 10, padding: '1rem 1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: 14, color: 'var(--warning)' },
  tips: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.25rem 1.5rem', display: 'flex', flexDirection: 'column', gap: 8 },
  tipsTitle: { fontSize: 14, fontWeight: 600, color: 'var(--text)', marginBottom: 4 },
  tipText: { fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6 },
  code: { fontFamily: 'var(--font-mono)', background: 'var(--surface-2)', padding: '2px 6px', borderRadius: 4, fontSize: 12, color: 'var(--accent)' },
}
