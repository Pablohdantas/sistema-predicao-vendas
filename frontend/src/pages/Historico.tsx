import { Link } from 'react-router-dom'
import { useState } from 'react'
import AtualizacaoHistorico from '../components/AtualizacaoHistorico'
import { useExecucao } from '../context/ExecucaoContext'

export default function Historico() {
  const { execucaoId } = useExecucao()
  const [atualizado, setAtualizado] = useState(false)

  return (
    <div className="fluxo-conteudo">
      <header className="page-header"><div>
        <p className="eyebrow">Acompanhamento</p>
        <h1>Atualização de vendas</h1>
        <p className="subtitle">Adicione novas vendas sem executar outro treinamento AutoML.</p>
      </div></header>
      <AtualizacaoHistorico
        key={execucaoId}
        execucaoId={execucaoId}
        treinamentoConcluido
        onHistoricoAtualizado={() => setAtualizado(true)}
      />
      {atualizado && (
        <section className="panel fluxo-acoes">
          <p>Histórico ampliado. Já é possível gerar previsões a partir dos dados mais recentes.</p>
          <Link to="/previsoes" className="fluxo-link-primario">Gerar nova previsão semanal</Link>
        </section>
      )}
    </div>
  )
}
