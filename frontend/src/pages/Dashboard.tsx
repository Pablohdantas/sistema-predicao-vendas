import { Link } from 'react-router-dom'
import { Activity, CalendarDays } from 'lucide-react'
import { useExecucao } from '../context/ExecucaoContext'

const formatar = (valor: number | undefined) =>
  valor == null ? '—' : new Intl.NumberFormat('pt-BR', { maximumFractionDigits: 2 }).format(valor)

export default function Dashboard() {
  const { execucaoId, statusSemanal } = useExecucao()
  const avaliacao = statusSemanal?.avaliacao_externa
  const horizontes = Object.entries(avaliacao?.por_horizonte ?? {})
    .sort(([a], [b]) => Number(a) - Number(b))

  return (
    <div className="fluxo-conteudo">
      <header className="page-header">
        <div>
          <p className="eyebrow">Análise de resultados</p>
          <h1>Visão geral</h1>
          <p className="subtitle">Acompanhamento do modelo selecionado para esta execução.</p>
        </div>
      </header>
      <section className="summary-grid">
        <div className="metric-card"><span className="metric-label">Modelo vencedor</span><strong className="metric-value">{statusSemanal?.algoritmo_vencedor?.toUpperCase() ?? '—'}</strong><span className="metric-hint">AutoML semanal</span></div>
        <div className="metric-card"><span className="metric-label">RMSE externo</span><strong className="metric-value">{formatar(avaliacao?.agregado.rmse)}</strong><span className="metric-hint">Menor é melhor</span></div>
        <div className="metric-card"><span className="metric-label">MAE externo</span><strong className="metric-value">{formatar(avaliacao?.agregado.mae)}</strong><span className="metric-hint">Erro absoluto médio</span></div>
        <div className="metric-card"><span className="metric-label">Previsões avaliadas</span><strong className="metric-value">{formatar(avaliacao?.agregado.quantidade)}</strong><span className="metric-hint">Teste reservado</span></div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <div><h2>Resultados por horizonte</h2><p>Erros calculados separadamente para cada dia de antecedência.</p></div>
          <Activity className="heading-icon" size={23} />
        </div>
        {horizontes.length ? (
          <div className="fluxo-tabela-rolagem">
            <table className="fluxo-tabela"><thead><tr><th>Horizonte</th><th>RMSE</th><th>MAE</th><th>Registros</th></tr></thead>
              <tbody>{horizontes.map(([dia, m]) => (
                <tr key={dia}><td>Dia {dia}</td><td>{formatar(m.rmse)}</td><td>{formatar(m.mae)}</td><td>{formatar(m.quantidade)}</td></tr>
              ))}</tbody>
            </table>
          </div>
        ) : <p>O relatório de métricas externas ainda não está disponível no status desta execução.</p>}
      </section>
      <section className="panel fluxo-acoes">
        <p className="execution-id">ID: {execucaoId}</p>
        <Link to="/previsoes" className="fluxo-link-primario"><CalendarDays size={18} /> Consultar os próximos 7 dias</Link>
      </section>
    </div>
  )
}
