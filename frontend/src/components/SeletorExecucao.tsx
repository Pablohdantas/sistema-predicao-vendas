import { useCallback, useEffect, useState } from 'react'
import { Database, LoaderCircle, RefreshCw } from 'lucide-react'
import { api, obterMensagemErro } from '../services/api'

type Execucao = {
  execucao_id: string
  nome_arquivo: string
  status: string
  algoritmo_vencedor: string | null
}

type RespostaExecucoes = { quantidade: number; execucoes: Execucao[] }
type Props = {
  execucaoSelecionada: string
  onSelecionar: (id: string) => void
}

const nomesStatus: Record<string, string> = {
  aguardando_treinamento: 'Aguardando análise',
  nao_iniciado: 'Aguardando análise',
  em_andamento: 'Em análise',
  concluido: 'Pronto',
  falhou: 'Falhou',
  interrompido: 'Interrompido',
}

export default function SeletorExecucao({ execucaoSelecionada, onSelecionar }: Props) {
  const [execucoes, setExecucoes] = useState<Execucao[]>([])
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState('')

  const buscarExecucoes = useCallback(async () => {
    setCarregando(true)
    setErro('')
    try {
      const resposta = await api.get<RespostaExecucoes>('/execucoes/')
      setExecucoes(resposta.data.execucoes)
    } catch (falha) {
      setErro(obterMensagemErro(falha))
    } finally {
      setCarregando(false)
    }
  }, [])

  useEffect(() => {
    void buscarExecucoes()
  }, [buscarExecucoes])

  return (
    <section className="panel" id="execucoes">
      <div className="panel-heading">
        <div>
          <h2>Meus históricos de vendas</h2>
          <p>Continue de onde parou ou consulte análises anteriores.</p>
        </div>
        <Database className="heading-icon" size={23} />
      </div>

      <div className="execution-picker">
        <label htmlFor="seletor-execucao">Histórico selecionado</label>
        <select
          id="seletor-execucao"
          value={execucaoSelecionada}
          onChange={(evento) => onSelecionar(evento.target.value)}
        >
          <option value="">Selecione um histórico</option>
          {execucoes.map((execucao) => (
            <option key={execucao.execucao_id} value={execucao.execucao_id}>
              {execucao.nome_arquivo}
              {' · '}
              {nomesStatus[execucao.status] ?? execucao.status}
              {execucao.status === 'concluido' && execucao.algoritmo_vencedor
                ? ` (${execucao.algoritmo_vencedor.toUpperCase()})`
                : ''}
              {' · '}
              {execucao.execucao_id.slice(0, 8)}
            </option>
          ))}
        </select>
      </div>

      <button
        type="button"
        className="secondary-button"
        onClick={() => void buscarExecucoes()}
        disabled={carregando}
      >
        {carregando
          ? <LoaderCircle className="spin" size={17} />
          : <RefreshCw size={17} />}
        {carregando ? 'Buscando históricos...' : 'Atualizar lista'}
      </button>

      {erro && <p className="inline-error" role="alert">{erro}</p>}
      {!carregando && !erro && execucoes.length === 0 && (
        <p className="empty-executions">Nenhum histórico encontrado. Envie seu primeiro CSV.</p>
      )}
      {execucoes.length > 0 && (
        <p className="execution-count">{execucoes.length} histórico(s) disponível(is).</p>
      )}
    </section>
  )
}
