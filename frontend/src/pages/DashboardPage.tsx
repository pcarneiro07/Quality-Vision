import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import { RefreshCw, Activity, Target, Zap } from 'lucide-react'
import { useTrainingLogs } from '../hooks/useTrainingLogs'
import { fetchEvaluationResults } from '../hooks/api'
import type { EvaluationResults } from '../types'

export function DashboardPage() {
  const { logs, isPolling, setIsPolling, error } = useTrainingLogs()
  const [evalResults, setEvalResults] = useState<EvaluationResults | null>(null)
  const [evalError, setEvalError] = useState<string | null>(null)

  useEffect(() => {
    fetchEvaluationResults()
      .then(setEvalResults)
      .catch(() => setEvalError('Execute evaluate.py para ver a matriz de confusão.'))
  }, [])

  const latest = logs[logs.length - 1]
  const bestValAcc = logs.length > 0 ? Math.max(...logs.map(l => l.val_acc)) : null
  const bestEpoch = bestValAcc ? logs.find(l => l.val_acc === bestValAcc)?.epoch : null

  return (
    <div style={styles.container}>
      <div style={styles.kpiRow}>
        <KpiCard
          icon={<Activity size={18} color="var(--accent)" />}
          label="Épocas Concluídas"
          value={logs.length.toString()}
          sub={isPolling ? '● Atualizando...' : '○ Pausado'}
          subColor={isPolling ? 'var(--accent)' : 'var(--text-muted)'}
        />
        <KpiCard
          icon={<Target size={18} color="var(--accent)" />}
          label="Melhor Val Acc"
          value={bestValAcc ? `${(bestValAcc * 100).toFixed(2)}%` : '—'}
          sub={bestEpoch ? `Época ${bestEpoch}` : ''}
        />
        <KpiCard
          icon={<Zap size={18} color="var(--accent)" />}
          label="Última Val Loss"
          value={latest ? latest.val_loss.toFixed(4) : '—'}
          sub={latest ? `LR: ${latest.lr.toExponential(2)}` : ''}
        />
        {evalResults && (
          <KpiCard
            icon={<Target size={18} color="var(--accent)" />}
            label="F1-Score (Teste)"
            value={`${(evalResults.f1_score * 100).toFixed(2)}%`}
            sub={`${evalResults.total_test_samples} amostras`}
          />
        )}
      </div>

      <div style={styles.pollRow}>
        <button
          style={{ ...styles.pollBtn, ...(isPolling ? styles.pollBtnActive : {}) }}
          onClick={() => setIsPolling(!isPolling)}
        >
          <RefreshCw size={14} style={isPolling ? styles.spinSlow : {}} />
          {isPolling ? 'Pausar atualização' : 'Retomar atualização'}
        </button>
        <span style={styles.pollHint}>Polling a cada 2s — acompanhe o treino ao vivo</span>
      </div>

      {error && <div style={styles.errorBanner}>{error}</div>}

      {logs.length === 0 && !error && (
        <div style={styles.emptyState}>
          <p style={styles.emptyTitle}>Nenhum dado de treinamento ainda</p>
          <p style={styles.emptySub}>Execute <code style={styles.code}>python -m backend.model.train</code> e os gráficos aparecerão aqui automaticamente.</p>
        </div>
      )}

      {logs.length > 0 && (
        <>
          <ChartCard title="Acurácia por Época" subtitle="Treino vs Validação">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={logs} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2028" />
                <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} label={{ value: 'Época', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
                <YAxis domain={[0.8, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} stroke="#6b7280" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                <Tooltip
                  contentStyle={{ background: '#111318', border: '1px solid #2a2d35', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number, name: string) => [`${(v * 100).toFixed(2)}%`, name === 'train_acc' ? 'Treino' : 'Validação']}
                  labelFormatter={(l) => `Época ${l}`}
                />
                <Legend formatter={(v) => v === 'train_acc' ? 'Treino' : 'Validação'} />
                <Line type="monotone" dataKey="train_acc" stroke="#00d4aa" strokeWidth={2} dot={false} activeDot={{ r: 5 }} />
                <Line type="monotone" dataKey="val_acc" stroke="#ffa502" strokeWidth={2} dot={false} activeDot={{ r: 5 }} strokeDasharray="5 3" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Loss por Época" subtitle="Binary Cross-Entropy — Treino vs Validação">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={logs} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2028" />
                <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} label={{ value: 'Época', position: 'insideBottom', offset: -2, fill: '#6b7280', fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                <Tooltip
                  contentStyle={{ background: '#111318', border: '1px solid #2a2d35', borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number, name: string) => [v.toFixed(4), name === 'train_loss' ? 'Treino' : 'Validação']}
                  labelFormatter={(l) => `Época ${l}`}
                />
                <Legend formatter={(v) => v === 'train_loss' ? 'Treino' : 'Validação'} />
                <Line type="monotone" dataKey="train_loss" stroke="#ff4757" strokeWidth={2} dot={false} activeDot={{ r: 5 }} />
                <Line type="monotone" dataKey="val_loss" stroke="#a78bfa" strokeWidth={2} dot={false} activeDot={{ r: 5 }} strokeDasharray="5 3" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </>
      )}

      {evalResults ? (
        <ChartCard title="Matriz de Confusão" subtitle="Conjunto de Teste">
          <ConfusionMatrixDisplay results={evalResults} />
        </ChartCard>
      ) : (
        <div style={styles.evalPlaceholder}>
          <p style={styles.evalPlaceholderText}>
            {evalError ?? 'Carregando matriz de confusão...'}
          </p>
        </div>
      )}
    </div>
  )
}

function KpiCard({ icon, label, value, sub, subColor }: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  subColor?: string
}) {
  return (
    <div style={kpiStyles.card}>
      <div style={kpiStyles.header}>
        {icon}
        <span style={kpiStyles.label}>{label}</span>
      </div>
      <span style={kpiStyles.value}>{value}</span>
      {sub && <span style={{ ...kpiStyles.sub, color: subColor ?? 'var(--text-muted)' }}>{sub}</span>}
    </div>
  )
}

const kpiStyles: Record<string, React.CSSProperties> = {
  card: { flex: 1, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12, padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: 6, minWidth: 140 },
  header: { display: 'flex', alignItems: 'center', gap: 8 },
  label: { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.05em' },
  value: { fontSize: 28, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text)', letterSpacing: '-0.02em' },
  sub: { fontSize: 11, fontFamily: 'var(--font-mono)' },
}

function ChartCard({ title, subtitle, children }: {
  title: string
  subtitle: string
  children: React.ReactNode
}) {
  return (
    <div style={chartStyles.card}>
      <div style={chartStyles.header}>
        <h3 style={chartStyles.title}>{title}</h3>
        <span style={chartStyles.subtitle}>{subtitle}</span>
      </div>
      {children}
    </div>
  )
}

const chartStyles: Record<string, React.CSSProperties> = {
  card: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 16, padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' },
  header: { display: 'flex', flexDirection: 'column', gap: 4 },
  title: { fontSize: 16, fontWeight: 600, color: 'var(--text)' },
  subtitle: { fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' },
}

function ConfusionMatrixDisplay({ results }: { results: EvaluationResults }) {
  const { true_negatives: tn, false_positives: fp, false_negatives: fn, true_positives: tp } = results.business_impact

  const cellData = [
    { label: 'Verdadeiro Negativo (TN)', value: tn, desc: 'Peças OK corretamente aprovadas', color: 'var(--accent)' },
    { label: 'Falso Positivo (FP)', value: fp, desc: 'Peças OK descartadas incorretamente', color: 'var(--warning)' },
    { label: 'Falso Negativo (FN) ⚠️', value: fn, desc: 'Defeitos NÃO detectados — risco crítico!', color: 'var(--danger)' },
    { label: 'Verdadeiro Positivo (TP)', value: tp, desc: 'Defeitos corretamente detectados', color: 'var(--accent)' },
  ]

  return (
    <div style={cmStyles.wrapper}>
      <div style={cmStyles.grid}>
        {cellData.map((cell, i) => (
          <div key={i} style={{ ...cmStyles.cell, borderColor: cell.color + '44' }}>
            <span style={{ ...cmStyles.cellValue, color: cell.color }}>{cell.value}</span>
            <span style={cmStyles.cellLabel}>{cell.label}</span>
            <span style={cmStyles.cellDesc}>{cell.desc}</span>
          </div>
        ))}
      </div>

      <div style={cmStyles.metricsRow}>
        {[
          { label: 'Acurácia', value: `${(results.accuracy * 100).toFixed(2)}%` },
          { label: 'Precisão', value: `${(results.precision * 100).toFixed(2)}%` },
          { label: 'Recall', value: `${(results.recall * 100).toFixed(2)}%` },
          { label: 'F1-Score', value: `${(results.f1_score * 100).toFixed(2)}%` },
        ].map(({ label, value }) => (
          <div key={label} style={cmStyles.metric}>
            <span style={cmStyles.metricLabel}>{label}</span>
            <span style={cmStyles.metricValue}>{value}</span>
          </div>
        ))}
      </div>

      <div style={cmStyles.note}>
        💡 <strong>Nota de negócio:</strong> FN (defeito aprovado) é mais custoso que FP (peça OK descartada). Priorizamos Recall.
      </div>
    </div>
  )
}

const cmStyles: Record<string, React.CSSProperties> = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: '1rem' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' },
  cell: { background: 'var(--surface-2)', border: '1px solid', borderRadius: 10, padding: '1rem', display: 'flex', flexDirection: 'column', gap: 4 },
  cellValue: { fontSize: 32, fontWeight: 700, fontFamily: 'var(--font-mono)' },
  cellLabel: { fontSize: 12, fontWeight: 600, color: 'var(--text)' },
  cellDesc: { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4 },
  metricsRow: { display: 'flex', gap: '0.75rem' },
  metric: { flex: 1, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 8, padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center', textAlign: 'center' },
  metricLabel: { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase' },
  metricValue: { fontSize: 20, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent)' },
  note: { background: 'rgba(255, 165, 2, 0.06)', border: '1px solid rgba(255, 165, 2, 0.2)', borderRadius: 8, padding: '0.875rem 1rem', fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.5 },
}

const styles: Record<string, React.CSSProperties> = {
  container: { maxWidth: 900, margin: '0 auto', padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' },
  kpiRow: { display: 'flex', gap: '0.75rem', flexWrap: 'wrap' },
  pollRow: { display: 'flex', alignItems: 'center', gap: '1rem' },
  pollBtn: { display: 'flex', alignItems: 'center', gap: 6, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-muted)', padding: '0.5rem 1rem', cursor: 'pointer', fontSize: 13, fontFamily: 'var(--font-mono)', transition: 'all 0.2s' },
  pollBtnActive: { borderColor: 'var(--accent)', color: 'var(--accent)' },
  spinSlow: { animation: 'spin 2s linear infinite' },
  pollHint: { fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' },
  errorBanner: { background: 'rgba(255, 71, 87, 0.08)', border: '1px solid rgba(255, 71, 87, 0.3)', borderRadius: 10, padding: '0.875rem 1.25rem', fontSize: 13, color: 'var(--danger)' },
  emptyState: { background: 'var(--surface)', border: '1px dashed var(--border)', borderRadius: 16, padding: '3rem 2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', textAlign: 'center' },
  emptyTitle: { fontSize: 18, fontWeight: 600, color: 'var(--text)' },
  emptySub: { fontSize: 14, color: 'var(--text-muted)', maxWidth: 480, lineHeight: 1.6 },
  code: { fontFamily: 'var(--font-mono)', background: 'var(--surface-2)', padding: '2px 6px', borderRadius: 4, fontSize: 13, color: 'var(--accent)' },
  evalPlaceholder: { background: 'var(--surface)', border: '1px dashed var(--border)', borderRadius: 16, padding: '2rem', textAlign: 'center' },
  evalPlaceholderText: { fontSize: 14, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' },
}
