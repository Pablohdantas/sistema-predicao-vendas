
import { useState } from 'react'
import { Link } from 'react-router-dom'
import AtualizacaoHistorico from '../components/AtualizacaoHistorico'
import { useExecucao } from '../context/useExecucao'

type ConteudoProps = {
  execucaoId: string
  treinamentoConcluido: boolean
}

function ConteudoAtualizacao({
  execucaoId,
  treinamentoConcluido,
}: ConteudoProps) {
  const [atualizado, setAtualizado] = useState(false)

  return (
    <>
      <AtualizacaoHistorico
        execucaoId={execucaoId}
        treinamentoConcluido={treinamentoConcluido}
        onHistoricoAtualizado={() => {
          setAtualizado(true)
        }}
      />

      {atualizado && (
        <section className="panel fluxo-acoes">
          <p>
            Histórico ampliado. Já é possível gerar previsões
            a partir dos dados mais recentes.
          </p>

          <Link
            to="/previsoes"
            className="fluxo-link-primario"
          >
            Gerar nova previsão semanal
          </Link>
        </section>
      )}
    </>
  )
}

export default function Historico() {
  const { execucaoId, statusSemanal } = useExecucao()

  const treinamentoConcluido =
    statusSemanal?.status === 'concluido'

  return (
    <div className="fluxo-conteudo">
      <header className="page-header">
        <div>
          <p className="eyebrow">Acompanhamento</p>

          <h1>Atualização de vendas</h1>

          <p className="subtitle">
            Adicione novas vendas sem executar outro
            treinamento AutoML.
          </p>
        </div>
      </header>

      <ConteudoAtualizacao
        key={execucaoId}
        execucaoId={execucaoId}
        treinamentoConcluido={treinamentoConcluido}
      />
    </div>
  )
}
