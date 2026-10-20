
import { useState } from 'react'
import type { ChangeEvent } from 'react'
import {
  CheckCircle2,
  FileUp,
  History,
  LoaderCircle,
  TriangleAlert,
} from 'lucide-react'

import { api, obterMensagemErro } from '../services/api'

type ResultadoAtualizacao = {
  execucao_id: string
  status: string
  registros_adicionados: number
  registros_totais: number
  quantidade_lojas: number
  ultimas_datas_por_loja: Record<string, string>
  modelo_retreinado: boolean
  avaliacao_externa_preservada: boolean
}

type Props = {
  execucaoId: string
  treinamentoConcluido: boolean
  onHistoricoAtualizado: () => void
}

export default function AtualizacaoHistorico({
  execucaoId,
  treinamentoConcluido,
  onHistoricoAtualizado,
}: Props) {
  const [arquivo, setArquivo] = useState<File | null>(null)
  const [enviando, setEnviando] = useState(false)
  const [erro, setErro] = useState('')
  const [resultado, setResultado] =
    useState<ResultadoAtualizacao | null>(null)

  function selecionarArquivo(evento: ChangeEvent<HTMLInputElement>) {
    setArquivo(evento.target.files?.[0] ?? null)
    setErro('')
    setResultado(null)
  }

  async function enviarAtualizacao() {
    if (!arquivo || !execucaoId || !treinamentoConcluido) {
      setErro('Selecione uma execução concluída e um arquivo CSV.')
      return
    }

    setEnviando(true)
    setErro('')
    setResultado(null)

    const formulario = new FormData()
    formulario.append('arquivo', arquivo)

    try {
      const resposta = await api.post<ResultadoAtualizacao>(
        `/historico/execucoes/${execucaoId}/atualizar`,
        formulario,
      )

      setResultado(resposta.data)
      setArquivo(null)
      onHistoricoAtualizado()
    } catch (erroCapturado) {
      setErro(obterMensagemErro(erroCapturado))
    } finally {
      setEnviando(false)
    }
  }

  return (
    <section className="panel historico-panel" id="atualizar-historico">
      <div className="panel-heading">
        <div>
          <h2>Atualizar histórico de vendas</h2>
          <p>
            Acrescente registros novos e continue usando o
            modelo treinado, sem repetir o AutoML.
          </p>
        </div>
        <History className="heading-icon" size={23} />
      </div>

      <div className="historico-instrucoes">
        <strong>Como preparar a atualização</strong>
        <p>
          Envie um CSV contendo apenas as novas vendas, com
          loja_id, data e vendas. Se o dataset original
          tiver outras colunas, mantenha também essas colunas
          e informe os valores necessários.
        </p>
        <p>
          As datas devem ser posteriores ao último registro
          de cada loja. Limite de 10 MB.
        </p>
      </div>

      <label className="historico-arquivo">
        <FileUp size={24} />
        <span>
          {arquivo ? arquivo.name : 'Selecionar CSV de novas vendas'}
        </span>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={selecionarArquivo}
          disabled={!treinamentoConcluido || enviando}
        />
      </label>

      <button
        className="primary-button"
        type="button"
        disabled={
          !treinamentoConcluido ||
          !execucaoId ||
          !arquivo ||
          enviando
        }
        onClick={() => void enviarAtualizacao()}
      >
        {enviando ? (
          <LoaderCircle className="spin" size={18} />
        ) : (
          <FileUp size={18} />
        )}
        {enviando ? 'Atualizando histórico...' : 'Adicionar novas vendas'}
      </button>

      {!treinamentoConcluido && (
        <p className="previsao-ajuda">
          Selecione uma execução com treinamento definitivo concluído.
        </p>
      )}

      {erro && (
        <div className="previsao-erro" role="alert">
          <TriangleAlert size={18} />
          <span>{erro}</span>
        </div>
      )}

      {resultado && (
        <div className="historico-resultado" role="status">
          <div className="historico-sucesso">
            <CheckCircle2 size={19} />
            <strong>Histórico atualizado com sucesso!</strong>
          </div>

          <p>
            <strong>{resultado.registros_adicionados}</strong> novo(s)
            registro(s) adicionado(s).
          </p>
          <p>
            Total armazenado: {resultado.registros_totais} registros.
          </p>
          <p>
            O modelo não foi retreinado. Os resultados da avaliação
            externa original foram preservados.
          </p>
          <p>
            Agora você pode gerar outra previsão com o histórico atualizado.
          </p>
        </div>
      )}
    </section>
  )
}
