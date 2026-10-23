import { useEffect, useState } from 'react'
import PrevisaoSemanal from '../components/PrevisaoSemanal'
import { useExecucao } from '../context/ExecucaoContext'
import { api, obterMensagemErro } from '../services/api'

type RespostaLojas = { lojas: number[] }

export default function Previsoes() {
  const { execucaoId } = useExecucao()
  const [lojas, setLojas] = useState<string[]>([])
  const [erro, setErro] = useState('')

  useEffect(() => {
    let ativo = true
    api.get<RespostaLojas>(`/previsoes/execucoes/${execucaoId}/lojas`)
      .then((resposta) => {
        if (ativo) setLojas(resposta.data.lojas.map(String))
      })
      .catch((falha: unknown) => {
        if (ativo) setErro(obterMensagemErro(falha))
      })
    return () => { ativo = false }
  }, [execucaoId])

  return (
    <div className="fluxo-conteudo">
      <header className="page-header"><div>
        <p className="eyebrow">Próximos sete dias</p>
        <h1>Previsões de vendas</h1>
        <p className="subtitle">Selecione a loja, gere previsões e consulte as explicações SHAP por dia.</p>
      </div></header>
      {erro && <p className="inline-error" role="alert">{erro}</p>}
      <PrevisaoSemanal key={execucaoId} execucaoId={execucaoId} lojas={lojas} treinamentoConcluido />
    </div>
  )
}
