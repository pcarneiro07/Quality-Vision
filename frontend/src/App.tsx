import { useState, useEffect } from 'react'
import { Cpu, BarChart2, Activity } from 'lucide-react'
import { InspectionPage } from './pages/InspectionPage'
import { DashboardPage } from './pages/DashboardPage'
import { fetchHealth } from './hooks/api'

type Tab = 'inspection' | 'dashboard'

export default function App() {
  const [tab, setTab] = useState<Tab>('inspection')
  const [apiStatus, setApiStatus] = useState<{ ok: boolean; modelLoaded: boolean; device: string } | null>(null)

  useEffect(() => {
    fetchHealth()
      .then((h) => setApiStatus({ ok: true, modelLoaded: h.model_loaded, device: h.device }))
      .catch(() => setApiStatus({ ok: false, modelLoaded: false, device: '' }))
    const interval = setInterval(() => {
      fetchHealth()
        .then((h) => setApiStatus({ ok: true, modelLoaded: h.model_loaded, device: h.device }))
        .catch(() => setApiStatus({ ok: false, modelLoaded: false, device: '' }))
    }, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div style={styles.root}>
      <aside style={styles.sidebar}>
        <div style={styles.logoArea}>
          <div style={styles.logoIcon}>
            <Cpu size={22} color="var(--accent)" strokeWidth={1.5} />
          </div>
          <div>
            <p style={styles.logoTitle}>Quality Vision</p>
            <p style={styles.logoSub}>Inspeção Industrial v1.0</p>
          </div>
        </div>

        <div style={styles.divider} />

        <nav style={styles.nav}>
          <NavItem
            icon={<Activity size={17} />}
            label="Inspeção"
            desc="Upload e predição"
            active={tab === 'inspection'}
            onClick={() => setTab('inspection')}
          />
          <NavItem
            icon={<BarChart2 size={17} />}
            label="Dashboard"
            desc="Métricas e curvas"
            active={tab === 'dashboard'}
            onClick={() => setTab('dashboard')}
          />
        </nav>

        <div style={styles.spacer} />

        <div style={styles.statusBox}>
          <div style={styles.statusDot(apiStatus?.ok ?? null)} />
          <div>
            <p style={styles.statusLabel}>
              {apiStatus === null ? 'Conectando...' : apiStatus.ok ? 'API Online' : 'API Offline'}
            </p>
            {apiStatus?.ok && (
              <p style={styles.statusSub}>
                {apiStatus.modelLoaded ? `Modelo ✓ | ${apiStatus.device}` : 'Modelo não carregado'}
              </p>
            )}
            {apiStatus && !apiStatus.ok && (
              <p style={styles.statusSub}>uvicorn na porta 8000</p>
            )}
          </div>
        </div>
      </aside>

      <main style={styles.main}>
        <div style={styles.pageHeader}>
          <h1 style={styles.pageTitle}>
            {tab === 'inspection' ? 'Inspeção de Peças' : 'Dashboard de Métricas'}
          </h1>
          <p style={styles.pageSub}>
            {tab === 'inspection'
              ? 'Faça upload de uma imagem para classificar a peça em tempo real'
              : 'Curvas de treinamento e avaliação do modelo'
            }
          </p>
        </div>

        <div style={styles.pageContent}>
          {tab === 'inspection' ? <InspectionPage /> : <DashboardPage />}
        </div>
      </main>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

function NavItem({ icon, label, desc, active, onClick }: {
  icon: React.ReactNode
  label: string
  desc: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button style={{ ...navStyles.item, ...(active ? navStyles.itemActive : {}) }} onClick={onClick}>
      <span style={{ ...navStyles.icon, ...(active ? navStyles.iconActive : {}) }}>
        {icon}
      </span>
      <div style={navStyles.text}>
        <span style={{ ...navStyles.label, ...(active ? navStyles.labelActive : {}) }}>
          {label}
        </span>
        <span style={navStyles.desc}>{desc}</span>
      </div>
      {active && <div style={navStyles.indicator} />}
    </button>
  )
}

const navStyles: Record<string, React.CSSProperties> = {
  item: { display: 'flex', alignItems: 'center', gap: '0.875rem', padding: '0.875rem 1rem', borderRadius: 10, border: 'none', background: 'transparent', cursor: 'pointer', transition: 'all 0.15s', width: '100%', textAlign: 'left', position: 'relative' },
  itemActive: { background: 'rgba(0, 212, 170, 0.08)', border: '1px solid rgba(0, 212, 170, 0.15)' },
  icon: { color: 'var(--text-muted)', display: 'flex', alignItems: 'center', flexShrink: 0 },
  iconActive: { color: 'var(--accent)' },
  text: { display: 'flex', flexDirection: 'column', gap: 2 },
  label: { fontSize: 14, fontWeight: 500, color: 'var(--text-muted)' },
  labelActive: { color: 'var(--text)' },
  desc: { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' },
  indicator: { position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)', width: 3, height: 20, background: 'var(--accent)', borderRadius: '2px 0 0 2px' },
}

const styles: Record<string, any> = {
  root: { display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg)' },
  sidebar: { width: 240, flexShrink: 0, background: 'var(--surface)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', padding: '1.25rem', gap: '0.5rem', overflowY: 'auto' },
  logoArea: { display: 'flex', alignItems: 'center', gap: '0.875rem', padding: '0.5rem 0.25rem' },
  logoIcon: { width: 40, height: 40, background: 'var(--accent-dim)', border: '1px solid rgba(0, 212, 170, 0.3)', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 },
  logoTitle: { fontSize: 14, fontWeight: 700, color: 'var(--text)', fontFamily: 'var(--font-mono)' },
  logoSub: { fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 },
  divider: { height: 1, background: 'var(--border)', margin: '0.5rem 0' },
  nav: { display: 'flex', flexDirection: 'column', gap: '0.25rem' },
  spacer: { flex: 1 },
  statusBox: { display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.875rem', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 10 },
  statusDot: (ok: boolean | null): React.CSSProperties => ({ width: 8, height: 8, borderRadius: '50%', flexShrink: 0, background: ok === null ? 'var(--text-muted)' : ok ? 'var(--accent)' : 'var(--danger)', boxShadow: ok ? '0 0 6px var(--accent)' : 'none' }),
  statusLabel: { fontSize: 12, fontWeight: 600, color: 'var(--text)', fontFamily: 'var(--font-mono)' },
  statusSub: { fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: 2 },
  main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  pageHeader: { padding: '1.5rem 2rem 1rem', borderBottom: '1px solid var(--border)', background: 'var(--surface)' },
  pageTitle: { fontSize: 22, fontWeight: 700, color: 'var(--text)', letterSpacing: '-0.02em' },
  pageSub: { fontSize: 13, color: 'var(--text-muted)', marginTop: 4, fontFamily: 'var(--font-mono)' },
  pageContent: { flex: 1, overflowY: 'auto', background: 'var(--bg)' },
}
