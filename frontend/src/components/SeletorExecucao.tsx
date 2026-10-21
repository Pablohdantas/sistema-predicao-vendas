
import { useEffect, useState } from 'react'
import {
  Database,
  LoaderCircle,
  RefreshCw,
} from 'lucide-react'

import { api, obterMensagemErro } from '../services/api'

type Execucao = {
  execucao_id: string
  nome_arquivo: string
  quantidade_linhas: number | null
  quantidade_lojas: number | null
  quantidade_dias: number | null
  status: string
  algoritmo_vencedor: string | null
}

type RespostaExecucoes = {
  quantidade: number
  execucoes: Execucao[]
}

type Props = {
  execucaoSelecionada: string
  onSelecionar: (id: string) => void
}

const nomesStatus: Record<string, string> = {
  aguardando_treinamento: 'Aguardando treinamento',
  em_andamento: 'Em treinamento',
  concluido: 'Concluído',
  falhou: 'Falhou',
  interrompido: 'Interrompido',
}

export default function SeletorExecucao({
  execucaoSelecionada,
  onSelecionar,
}: Props) {
  const [execucoes, setExecucoes] = useState<Execucao[]>([])
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState('')

  async function buscarExecucoes() {
    setCarregando(true)
    setErro('')

    try {
      const resposta = await api.get<RespostaExecucoes>(
        '/execucoes/',
      )

      setExecucoes(resposta.data.execucoes)
    } catch (erroCapturado) {
      setErro(obterMensagemErro(erroCapturado))
    } finally {
      setCarregando(false)
    }
  }

  useEffect(() => {
    // Executa a primeira consulta após montar o componente.
    const tarefa = window.setTimeout(() => {
      void buscarExecucoes()
    }, 0)

    return () => window.clearTimeout(tarefa)
  }, [])

  return (
    <section className="panel" id="execucoes">
      <div className="panel-heading">
        <div>
          <h2>Minhas execuções</h2>
          <p>
            Selecione um dataset enviado anteriormente
            para consultar seu treinamento e resultados.
          </p>
        </div>
        <Database className="heading-icon" size={23} />
      </div>

      <div className="execution-picker">
        <label htmlFor="seletor-execucao">
          Execução selecionada
        </label>

        <select
          id="seletor-execucao"
          value={execucaoSelecionada}
          onChange={(evento) => onSelecionar(evento.target.value)}
        >
          <option value="">Selecione uma execução</option>

          {execucoes.map((execucao) => (
            <option
              key={execucao.execucao_id}
              value={execucao.execucao_id}
            >
              {execucao.nome_arquivo}
              {' · '}
              {nomesStatus[execucao.status] ?? execucao.status}
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
        {carregando
          ? 'Buscando execuções...'
          : 'Atualizar lista'}
      </button>

      {erro && (
        <p className="inline-error" role="alert">
          {erro}
        </p>
      )}

      {!carregando && !erro && execucoes.length === 0 && (
        <p className="empty-executions">
          Nenhuma execução encontrada. Envie seu primeiro CSV.
        </p>
      )}

      {execucoes.length > 0 && (
        <p className="execution-count">
          {execucoes.length} execução(ões) disponível(is).
        </p>
      )}
    </section>
  )
}
