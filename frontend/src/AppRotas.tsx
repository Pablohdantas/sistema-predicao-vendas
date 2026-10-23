import { BrowserRouter, Navigate, NavLink, Outlet, Route, Routes } from 'react-router-dom'
import {
  BarChart3,
  BrainCircuit,
  CalendarDays,
  Database,
  History,
  LoaderCircle,
} from 'lucide-react'
import { ExecucaoProvider } from './context/ExecucaoContext'
import { useExecucao } from './context/useExecucao'
import Entrada from './pages/Entrada'
import Treinamento from './pages/Treinamento'
import Dashboard from './pages/Dashboard'
import Previsoes from './pages/Previsoes'
import Historico from './pages/Historico'
import './App.css'
import './fluxo.css'

function Carregando() {
  return (
    <div className="fluxo-carregando" role="status">
      <LoaderCircle size={22} className="spin" />
      Verificando o histórico de vendas...
    </div>
  )
}

function ProtegerDataset() {
  const { carregando, execucaoId } = useExecucao()
  if (carregando) return <Carregando />
  return execucaoId ? <Outlet /> : <Navigate to="/entrada" replace />
}

function ProtegerModelo() {
  const { carregando, execucaoId, statusSemanal } = useExecucao()
  if (carregando) return <Carregando />
  if (!execucaoId) return <Navigate to="/entrada" replace />
  return statusSemanal?.status === 'concluido'
    ? <Outlet />
    : <Navigate to="/treinamento" replace />
}

function Inicio() {
  const { carregando, execucaoId, statusSemanal } = useExecucao()
  if (carregando) return <Carregando />
  if (!execucaoId) return <Navigate to="/entrada" replace />
  if (statusSemanal?.status === 'concluido') {
    return <Navigate to="/dashboard" replace />
  }
  return <Navigate to="/treinamento" replace />
}

function Estrutura() {
  const { execucaoId, statusSemanal, erro } = useExecucao()
  const modeloDisponivel = Boolean(execucaoId) && statusSemanal?.status === 'concluido'
  const linkClasse = ({ isActive }: { isActive: boolean }) => isActive ? 'ativo' : ''

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-symbol"><BrainCircuit size={23} /></div>
          <div><strong>Previsão de Vendas</strong><span>AutoML + XAI</span></div>
        </div>
        <nav className="navigation" aria-label="Navegação principal">
          <NavLink to="/entrada" className={linkClasse}>
            <Database size={19} /> Dados
          </NavLink>
          {modeloDisponivel && (
            <>
              <NavLink to="/dashboard" className={linkClasse}>
                <BarChart3 size={19} /> Visão geral
              </NavLink>
              <NavLink to="/previsoes" className={linkClasse}>
                <CalendarDays size={19} /> Previsões
              </NavLink>
              <NavLink to="/historico" className={linkClasse}>
                <History size={19} /> Atualizar vendas
              </NavLink>
            </>
          )}
        </nav>
        <div className="sidebar-footer"><span className="status-dot" /> AutoML + XAI</div>
      </aside>
      <main className="main-content">
        {erro && <div className="alert alert-error" role="alert">{erro}</div>}
        <Outlet />
        <footer className="page-footer">Sistema de Predição de Vendas · AutoML + XAI</footer>
      </main>
    </div>
  )
}

export default function AppRotas() {
  return (
    <BrowserRouter>
      <ExecucaoProvider>
        <Routes>
          <Route element={<Estrutura />}>
            <Route index element={<Inicio />} />
            <Route path="entrada" element={<Entrada />} />
            <Route element={<ProtegerDataset />}>
              {/* Rota interna: treinamento automático sem item no menu. */}
              <Route path="treinamento" element={<Treinamento />} />
            </Route>
            <Route element={<ProtegerModelo />}>
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="previsoes" element={<Previsoes />} />
              <Route path="historico" element={<Historico />} />
            </Route>
            <Route path="*" element={<Inicio />} />
          </Route>
        </Routes>
      </ExecucaoProvider>
    </BrowserRouter>
  )
}
