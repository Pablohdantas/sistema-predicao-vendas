
import { useCallback, useEffect, useState } from 'react'
import type { ChangeEvent } from 'react'

import {
  Activity,
  BarChart3,
  BrainCircuit,
  CalendarDays,
  CheckCircle2,
  Clock3,
  CloudUpload,
  Database,
  FileSpreadsheet,
  History,
  LoaderCircle,
  Play,
  RefreshCw,
  Store,
  TriangleAlert,
} from 'lucide-react'

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { api, obterMensagemErro } from './services/api'

import type {
  AvaliacaoExterna,
  DatasetResponse,
  ListaPrevisoes,
  PrevisaoExterna,
  StatusTreinamento,
} from './services/api'

import GuiaTemplate from './components/GuiaTemplate'
import ConfiguracaoAutoML from './components/ConfiguracaoAutoML'
import SeletorExecucao from './components/SeletorExecucao'
import PrevisaoFutura from './components/PrevisaoFutura'
import PrevisaoSemanal from './components/PrevisaoSemanal'
import TreinamentoSemanal from './components/TreinamentoSemanal'
import AtualizacaoHistorico from './components/AtualizacaoHistorico'

import {
  parametrosValidos,
} from './utils/configuracaoAutoML'

import type {
  ParametrosTreinamento,
} from './utils/configuracaoAutoML'

import './App.css'

const CHAVE_EXECUCAO = 'automl_execucao_atual'

type RespostaLojas = {
  execucao_id: string
  quantidade_lojas: number
  lojas: number[]
}

type ModeloSemanalDisponivel = {
  execucaoId: string
  disponivel: boolean
}

const formatarNumero = (
  valor: number,
  casas = 2,
): string => {
  return new Intl.NumberFormat('pt-BR', {
    minimumFractionDigits: casas,
    maximumFractionDigits: casas,
  }).format(valor)
}

const formatarData = (data: string): string => {
  const [ano, mes, dia] = data.slice(0, 10).split('-')

  return `${dia}/${mes}/${ano}`
}

const etiquetaStatus: Record<string, string> = {
  aguardando_treinamento: 'Aguardando treinamento',
  em_andamento: 'Treinando',
  concluido: 'Concluído',
  falhou: 'Falhou',
  interrompido: 'Interrompido',
}

function App() {
  // --------------------------------------------------
  // Dataset e execução selecionada
  // --------------------------------------------------

  const [arquivo, setArquivo] = useState<File | null>(null)

  const [execucaoId, setExecucaoId] = useState<string>(() => {
    return localStorage.getItem(CHAVE_EXECUCAO) ?? ''
  })

  const [upload, setUpload] =
    useState<DatasetResponse | null>(null)

  const [status, setStatus] =
    useState<StatusTreinamento | null>(null)

  const [avaliacao, setAvaliacao] =
    useState<AvaliacaoExterna | null>(null)

  const [previsoes, setPrevisoes] =
    useState<PrevisaoExterna[]>([])

  const [lojaSelecionada, setLojaSelecionada] =
    useState<string>('')

  const [lojasExecucao, setLojasExecucao] =
    useState<string[]>([])

  const [versaoListaExecucoes, setVersaoListaExecucoes] =
    useState(0)

  const [versaoHistorico, setVersaoHistorico] =
    useState(0)

  // O modelo semanal pertence a uma execução específica.
  // Guardamos o ID para nunca reutilizar indevidamente
  // o estado de uma execução anterior.
  const [modeloSemanal, setModeloSemanal] =
    useState<ModeloSemanalDisponivel | null>(null)

  // --------------------------------------------------
  // Carregamento e mensagens
  // --------------------------------------------------

  const [enviando, setEnviando] = useState(false)
  const [iniciando, setIniciando] = useState(false)

  const [
    carregandoResultados,
    setCarregandoResultados,
  ] = useState(false)

  const [
    atualizandoStatusManual,
    setAtualizandoStatusManual,
  ] = useState(false)

  const [mensagemErro, setMensagemErro] = useState('')
  const [mensagemSucesso, setMensagemSucesso] = useState('')

  // --------------------------------------------------
  // Configuração do treinamento convencional
  // --------------------------------------------------

  const [
    configuracaoTreinamento,
    setConfiguracaoTreinamento,
  ] = useState<ParametrosTreinamento>({
    modo_teste: false,
    tentativas: 2,
    tempo_por_algoritmo: 60,
  })

  // --------------------------------------------------
  // Disponibilidade do modelo semanal
  // --------------------------------------------------

  const atualizarDisponibilidadeSemanal = useCallback(
    (disponivel: boolean) => {
      setModeloSemanal((anterior) => {
        if (
          anterior?.execucaoId === execucaoId &&
          anterior.disponivel === disponivel
        ) {
          return anterior
        }

        return {
          execucaoId,
          disponivel,
        }
      })
    },
    [execucaoId],
  )

  const treinamentoSemanalConcluido =
    modeloSemanal?.execucaoId === execucaoId &&
    modeloSemanal.disponivel

  // --------------------------------------------------
  // Consultar status do treinamento convencional
  // --------------------------------------------------

  const atualizarStatus = useCallback(
    async (id: string) => {
      const resposta = await api.get<StatusTreinamento>(
        `/treinamentos/${id}`,
      )

      setStatus(resposta.data)

      return resposta.data
    },
    [],
  )

  const atualizarStatusManualmente = async () => {
    if (!execucaoId || atualizandoStatusManual) {
      return
    }

    setAtualizandoStatusManual(true)
    setMensagemErro('')
    setMensagemSucesso('')

    try {
      const resultado = await atualizarStatus(execucaoId)

      const descricao =
        resultado.status === 'concluido'
          ? 'O treinamento está concluído.'
          : resultado.status === 'em_andamento'
            ? 'O treinamento está em andamento.'
            : resultado.status === 'falhou'
              ? 'O treinamento falhou.'
              : resultado.status === 'interrompido'
                ? 'O treinamento foi interrompido.'
                : 'O treinamento ainda não foi iniciado.'

      setMensagemSucesso(
        `Status atualizado com sucesso. ${descricao}`,
      )
    } catch (erro) {
      setMensagemErro(obterMensagemErro(erro))
    } finally {
      setAtualizandoStatusManual(false)
    }
  }

  // --------------------------------------------------
  // Carregar avaliação externa convencional
  // --------------------------------------------------

  const carregarResultados = useCallback(
    async (id: string) => {
      setCarregandoResultados(true)

      try {
        const [respostaAvaliacao, respostaPrevisoes] =
          await Promise.all([
            api.get<AvaliacaoExterna>(
              `/avaliacoes/${id}`,
            ),

            api.get<ListaPrevisoes>(
              `/avaliacoes/${id}/previsoes`,
              {
                params: {
                  limite: 1000,
                },
              },
            ),
          ])

        setAvaliacao(respostaAvaliacao.data)

        setPrevisoes(
          respostaPrevisoes.data.previsoes,
        )
      } catch (erro) {
        setMensagemErro(obterMensagemErro(erro))
      } finally {
        setCarregandoResultados(false)
      }
    },
    [],
  )

  // --------------------------------------------------
  // Buscar status da execução selecionada
  // --------------------------------------------------

  useEffect(() => {
    if (!execucaoId) {
      return
    }

    let cancelado = false

    const consultar = async () => {
      try {
        const resposta = await api.get<StatusTreinamento>(
          `/treinamentos/${execucaoId}`,
        )

        if (!cancelado) {
          setStatus(resposta.data)
        }
      } catch (erro) {
        if (!cancelado) {
          setMensagemErro(obterMensagemErro(erro))
        }
      }
    }

    void consultar()

    return () => {
      cancelado = true
    }
  }, [execucaoId])

  // --------------------------------------------------
  // Acompanhar treinamento convencional
  // --------------------------------------------------

  useEffect(() => {
    if (
      !execucaoId ||
      !status ||
      status.status !== 'em_andamento'
    ) {
      return
    }

    const intervalo = window.setInterval(() => {
      void atualizarStatus(execucaoId).catch(
        (erro: unknown) => {
          setMensagemErro(obterMensagemErro(erro))
        },
      )
    }, 2500)

    return () => {
      window.clearInterval(intervalo)
    }
  }, [
    execucaoId,
    status,
    atualizarStatus,
  ])

  // --------------------------------------------------
  // Buscar resultados convencionais concluídos
  // --------------------------------------------------

  useEffect(() => {
    if (
      !execucaoId ||
      status?.status !== 'concluido' ||
      status.modo !== 'treinamento_definitivo'
    ) {
      return
    }

    const tarefa = window.setTimeout(() => {
      void carregarResultados(execucaoId)
    }, 0)

    return () => {
      window.clearTimeout(tarefa)
    }
  }, [
    execucaoId,
    status?.status,
    status?.modo,
    carregarResultados,
  ])

  // --------------------------------------------------
  // Carregar lojas do histórico
  // --------------------------------------------------

  useEffect(() => {
    if (!execucaoId) {
      return
    }

    let cancelado = false

    const carregarLojas = async () => {
      try {
        const resposta = await api.get<RespostaLojas>(
          `/previsoes/execucoes/${execucaoId}/lojas`,
        )

        if (!cancelado) {
          const lojas = resposta.data.lojas
            .map(String)
            .sort((a, b) => Number(a) - Number(b))

          setLojasExecucao(lojas)
        }
      } catch (erro) {
        if (!cancelado) {
          setLojasExecucao([])
          setMensagemErro(obterMensagemErro(erro))
        }
      }
    }

    void carregarLojas()

    return () => {
      cancelado = true
    }
  }, [
    execucaoId,
    versaoHistorico,
  ])

  // --------------------------------------------------
  // Selecionar arquivo
  // --------------------------------------------------

  const selecionarArquivo = (
    evento: ChangeEvent<HTMLInputElement>,
  ) => {
    const selecionado =
      evento.target.files?.[0] ?? null

    setArquivo(selecionado)
    setMensagemErro('')
    setMensagemSucesso('')
  }

  // --------------------------------------------------
  // Enviar dataset
  // --------------------------------------------------

  const enviarDataset = async () => {
    if (!arquivo) {
      setMensagemErro(
        'Selecione um arquivo CSV primeiro.',
      )
      return
    }

    setEnviando(true)
    setMensagemErro('')
    setMensagemSucesso('')

    const formulario = new FormData()

    formulario.append('arquivo', arquivo)

    try {
      const resposta = await api.post<DatasetResponse>(
        '/datasets/upload',
        formulario,
      )

      const dados = resposta.data

      setUpload(dados)

      if (!dados.valido || !dados.execucao_id) {
        setMensagemErro(
          dados.erros.join(' | ') ||
          'O dataset não foi aprovado na validação.',
        )

        return
      }

      localStorage.setItem(
        CHAVE_EXECUCAO,
        dados.execucao_id,
      )

      setExecucaoId(dados.execucao_id)

      // Nova execução: todos os estados anteriores
      // deixam de ser válidos.
      setStatus(null)
      setAvaliacao(null)
      setPrevisoes([])
      setLojasExecucao([])
      setLojaSelecionada('')
      setModeloSemanal(null)

      setVersaoHistorico(0)

      setVersaoListaExecucoes(
        (versao) => versao + 1,
      )

      setMensagemSucesso(
        'Dataset validado e armazenado. ' +
        'Você já pode iniciar o AutoML semanal.',
      )
    } catch (erro) {
      setMensagemErro(obterMensagemErro(erro))
    } finally {
      setEnviando(false)
    }
  }

  // --------------------------------------------------
  // Iniciar treinamento convencional
  // --------------------------------------------------

  const iniciarTreinamento = async () => {
    if (
      !execucaoId ||
      !parametrosValidos(configuracaoTreinamento)
    ) {
      return
    }

    setIniciando(true)
    setMensagemErro('')
    setMensagemSucesso('')

    try {
      const resposta = await api.post<StatusTreinamento>(
        `/treinamentos/${execucaoId}`,
        configuracaoTreinamento,
      )

      setStatus(resposta.data)

      setMensagemSucesso(
        'Treinamento convencional iniciado!',
      )
    } catch (erro) {
      setMensagemErro(obterMensagemErro(erro))
    } finally {
      setIniciando(false)
    }
  }

  // --------------------------------------------------
  // Lojas e gráfico de avaliação
  // --------------------------------------------------

  const lojasAvaliacao = Array.from(
    new Set(
      previsoes.map((item) => String(item.loja_id)),
    ),
  ).sort((a, b) => Number(a) - Number(b))

  const lojasDisponiveis = lojasExecucao

  const lojaAtiva = lojasAvaliacao.includes(lojaSelecionada)
    ? lojaSelecionada
    : lojasAvaliacao[0] || ''

  const dadosGrafico = previsoes
    .filter(
      (item) => String(item.loja_id) === lojaAtiva,
    )
    .map((item) => ({
      data: formatarData(item.data_prevista),
      real: item.vendas_reais,
      prevista: item.vendas_previstas,
    }))

  // --------------------------------------------------
  // Estado do treinamento convencional
  // --------------------------------------------------

  const emAndamento =
    status?.status === 'em_andamento'

  const concluido =
    status?.status === 'concluido'

  const treinamentoDefinitivoConcluido =
    status?.status === 'concluido' &&
    status?.modo === 'treinamento_definitivo'

  const podeTreinar =
    Boolean(execucaoId) &&
    !emAndamento &&
    !iniciando &&
    status?.status !== 'concluido'

  // --------------------------------------------------
  // Trocar execução selecionada
  // --------------------------------------------------

  const selecionarExecucao = (id: string) => {
    if (id === execucaoId) {
      return
    }

    if (id) {
      localStorage.setItem(CHAVE_EXECUCAO, id)
    } else {
      localStorage.removeItem(CHAVE_EXECUCAO)
    }

    setExecucaoId(id)

    setArquivo(null)
    setUpload(null)
    setStatus(null)
    setAvaliacao(null)
    setPrevisoes([])
    setLojasExecucao([])
    setLojaSelecionada('')
    setModeloSemanal(null)

    setVersaoHistorico(0)

    setMensagemErro('')
    setMensagemSucesso('')
  }

  // --------------------------------------------------
  // Interface
  // --------------------------------------------------

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-symbol">
            <BrainCircuit size={23} />
          </div>

          <div>
            <strong>Previsão de Vendas</strong>
            <span>AutoML + XAI</span>
          </div>
        </div>

        <nav
          className="navigation"
          aria-label="Navegação principal"
        >
          <a href="#execucoes">
            <Database size={19} />
            Execuções
          </a>

          <a href="#previsao-semanal">
            <CalendarDays size={19} />
            Previsões futuras
          </a>

          <a href="#atualizar-historico">
            <History size={19} />
            Atualizar vendas
          </a>

          <a href="#visao-geral">
            <BarChart3 size={19} />
            Visão geral
          </a>

          <a href="#dataset">
            <FileSpreadsheet size={19} />
            Dataset
          </a>

          <a href="#treinamento-semanal">
            <BrainCircuit size={19} />
            Treinamento
          </a>

          <a href="#avaliacao">
            <Activity size={19} />
            Avaliação
          </a>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          Protótipo acadêmico
        </div>
      </aside>

      <main
        className="main-content"
        id="visao-geral"
      >
        <header className="page-header">
          <div>
            <p className="eyebrow">
              Plataforma de inteligência de vendas
            </p>

            <h1>Dashboard</h1>

            <p className="subtitle">
              Envie seu histórico, treine modelos e
              acompanhe o desempenho das previsões.
            </p>
          </div>

          <div className="header-tag">
            <Activity size={17} />
            AutoML
          </div>
        </header>

        {/* Mensagens */}

        {mensagemErro && (
          <div
            className="alert alert-error"
            role="alert"
          >
            <TriangleAlert size={20} />

            <span>{mensagemErro}</span>

            <button
              type="button"
              onClick={() => setMensagemErro('')}
              aria-label="Fechar erro"
            >
              ×
            </button>
          </div>
        )}

        {mensagemSucesso && (
          <div
            className="alert alert-success"
            role="status"
          >
            <CheckCircle2 size={20} />
            <span>{mensagemSucesso}</span>
          </div>
        )}

        {/* Métricas do modelo convencional */}

        <section
          className="summary-grid"
          aria-label="Resumo do modelo convencional"
        >
          <div className="metric-card">
            <span className="metric-label">
              Modelo convencional
            </span>

            <strong className="metric-value">
              {status?.resultado
                ?.algoritmo_vencedor
                ?.toUpperCase() ?? '—'}
            </strong>

            <span className="metric-hint">
              Horizonte de 7 dias
            </span>
          </div>

          <div className="metric-card">
            <span className="metric-label">
              RMSE externo
            </span>

            <strong className="metric-value">
              {avaliacao
                ? formatarNumero(avaliacao.rmse)
                : '—'}
            </strong>

            <span className="metric-hint">
              Avaliação convencional
            </span>
          </div>

          <div className="metric-card">
            <span className="metric-label">
              MAE externo
            </span>

            <strong className="metric-value">
              {avaliacao
                ? formatarNumero(avaliacao.mae)
                : '—'}
            </strong>

            <span className="metric-hint">
              Erro absoluto médio
            </span>
          </div>

          <div className="metric-card">
            <span className="metric-label">
              Previsões avaliadas
            </span>

            <strong className="metric-value">
              {avaliacao
                ? formatarNumero(
                    avaliacao.quantidade_previsoes,
                    0,
                  )
                : '—'}
            </strong>

            <span className="metric-hint">
              Teste temporal convencional
            </span>
          </div>
        </section>

        {/* Minhas execuções */}

        <SeletorExecucao
          key={versaoListaExecucoes}
          execucaoSelecionada={execucaoId}
          onSelecionar={selecionarExecucao}
        />

        <div className="section-grid">
          {/* Upload */}

          <section
            className="panel"
            id="dataset"
          >
            <div className="panel-heading">
              <div>
                <h2>Enviar dataset</h2>

                <p>
                  Arquivo CSV com histórico de vendas.
                </p>
              </div>

              <CloudUpload
                className="heading-icon"
                size={23}
              />
            </div>

            <label className="upload-area">
              <CloudUpload size={30} />

              <strong>
                {arquivo
                  ? arquivo.name
                  : 'Selecione seu arquivo CSV'}
              </strong>

              <span>
                Histórico a partir de 90 dias ·
                limite de 100 MB
              </span>

              <input
                type="file"
                accept=".csv,text/csv"
                onChange={selecionarArquivo}
              />
            </label>

            <button
              className="primary-button"
              type="button"
              onClick={() => void enviarDataset()}
              disabled={enviando || !arquivo}
            >
              {enviando ? (
                <LoaderCircle
                  className="spin"
                  size={18}
                />
              ) : (
                <CloudUpload size={18} />
              )}

              {enviando
                ? 'Validando CSV...'
                : 'Enviar e validar'}
            </button>

            <GuiaTemplate />

            {upload?.valido && (
              <div className="dataset-details">
                <CheckCircle2 size={18} />

                <div>
                  <strong>
                    Dataset validado
                  </strong>

                  <p>
                    {formatarNumero(
                      upload.quantidade_linhas,
                      0,
                    )}{' '}
                    registros

                    {upload.quantidade_lojas != null &&
                      ` · ${upload.quantidade_lojas} lojas`}

                    {upload.quantidade_dias != null &&
                      ` · ${upload.quantidade_dias} dias`}
                  </p>
                </div>
              </div>
            )}

            {!!upload?.avisos?.length && (
              <div className="warnings">
                {upload.avisos.map(
                  (aviso, indice) => (
                    <p key={indice}>
                      {aviso}
                    </p>
                  ),
                )}
              </div>
            )}
          </section>

          {/* Treinamento convencional preservado */}

          <section
            className="panel"
            id="treinamento"
          >
            <div className="panel-heading">
              <div>
                <h2>Treinamento convencional</h2>

                <p>
                  Modelo de horizonte fixo de sete dias.
                </p>
              </div>

              <BrainCircuit
                className="heading-icon"
                size={23}
              />
            </div>

            <div className="training-status">
              <div className="training-status-top">
                <span>
                  Status da execução convencional
                </span>

                <span
                  className={`status-pill ${
                    status?.status ??
                    'aguardando_treinamento'
                  }`}
                >
                  {etiquetaStatus[
                    status?.status ?? ''
                  ] ?? 'Sem treinamento'}
                </span>
              </div>

              <strong>
                {status?.etapa?.replaceAll('_', ' ') ??
                  'Aguardando treinamento'}
              </strong>

              {execucaoId && (
                <p
                  className="execution-id"
                  title={execucaoId}
                >
                  ID: {execucaoId}
                </p>
              )}
            </div>

            <div className="training-info">
              <div>
                <Clock3 size={18} />

                <span>
                  Validação temporal adaptativa
                </span>
              </div>

              <div>
                <CheckCircle2 size={18} />

                <span>
                  Seleção pelo menor RMSE
                </span>
              </div>

              <div>
                <Activity size={18} />

                <span>
                  Avaliação externa automática
                </span>
              </div>
            </div>

            <ConfiguracaoAutoML
              valor={configuracaoTreinamento}
              onChange={setConfiguracaoTreinamento}
              disabled={
                emAndamento ||
                iniciando ||
                concluido
              }
            />

            <button
              className="primary-button"
              type="button"
              disabled={
                !podeTreinar ||
                !parametrosValidos(
                  configuracaoTreinamento,
                )
              }
              onClick={() =>
                void iniciarTreinamento()
              }
            >
              {emAndamento || iniciando ? (
                <LoaderCircle
                  className="spin"
                  size={18}
                />
              ) : (
                <Play size={18} />
              )}

              {emAndamento
                ? 'Treinamento em andamento'
                : iniciando
                  ? 'Iniciando...'
                  : concluido
                    ? 'Treinamento concluído'
                    : 'Iniciar treinamento convencional'}
            </button>

            {execucaoId && (
              <button
                type="button"
                className="secondary-button"
                disabled={atualizandoStatusManual}
                onClick={() =>
                  void atualizarStatusManualmente()
                }
              >
                {atualizandoStatusManual ? (
                  <LoaderCircle
                    className="spin"
                    size={17}
                  />
                ) : (
                  <RefreshCw size={17} />
                )}

                {atualizandoStatusManual
                  ? 'Consultando status...'
                  : 'Atualizar status'}
              </button>
            )}

            {status?.erro && (
              <p className="inline-error">
                {status.erro}
              </p>
            )}
          </section>
        </div>

        {/* Treinamento AutoML semanal principal */}

        <TreinamentoSemanal
          key={`treino-semanal-${execucaoId}`}
          execucaoId={execucaoId}
          onModeloDisponivel={
            atualizarDisponibilidadeSemanal
          }
        />

        {/* Avaliação convencional */}

        <section
          className="panel chart-panel"
          id="avaliacao"
        >
          <div className="panel-heading">
            <div>
              <h2>
                Vendas reais × previstas
              </h2>

              <p>
                Comparação no teste temporal externo
                do modelo convencional.
              </p>
            </div>

            {lojasAvaliacao.length > 0 && (
              <label className="store-selector">
                <Store size={18} />

                <span>Loja</span>

                <select
                  value={lojaAtiva}
                  onChange={(evento) =>
                    setLojaSelecionada(
                      evento.target.value,
                    )
                  }
                >
                  {lojasAvaliacao.map(
                    (loja) => (
                      <option
                        key={loja}
                        value={loja}
                      >
                        {loja}
                      </option>
                    ),
                  )}
                </select>
              </label>
            )}
          </div>

          {dadosGrafico.length > 0 ? (
            <>
              <div className="chart-container">
                <ResponsiveContainer
                  width="100%"
                  height="100%"
                >
                  <LineChart
                    data={dadosGrafico}
                    margin={{
                      top: 12,
                      right: 14,
                      left: 0,
                      bottom: 0,
                    }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                    />

                    <XAxis
                      dataKey="data"
                      tick={{
                        fontSize: 12,
                      }}
                    />

                    <YAxis
                      tick={{
                        fontSize: 12,
                      }}
                      width={70}
                    />

                    <Tooltip
                      formatter={(valor) =>
                        formatarNumero(
                          Number(valor),
                        )
                      }
                    />

                    <Legend />

                    <Line
                      name="Vendas reais"
                      dataKey="real"
                      stroke="#17846b"
                      strokeWidth={3}
                      dot={false}
                      type="monotone"
                    />

                    <Line
                      name="Vendas previstas"
                      dataKey="prevista"
                      stroke="#e5a449"
                      strokeWidth={3}
                      dot={false}
                      type="monotone"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <p className="chart-footnote">
                O gráfico mostra somente as
                observações recuperadas pela API
                (limite de 1.000).
                As métricas exibidas no topo
                consideram todo o teste externo
                convencional.
              </p>
            </>
          ) : (
            <div className="empty-chart">
              {carregandoResultados ? (
                <LoaderCircle
                  className="spin"
                  size={32}
                />
              ) : (
                <BarChart3 size={36} />
              )}

              <strong>
                {carregandoResultados
                  ? 'Carregando resultados...'
                  : 'Sem avaliação convencional'}
              </strong>

              <p>
                A avaliação convencional será exibida
                quando houver um modelo de horizonte
                fixo concluído. O AutoML semanal
                funciona independentemente dela.
              </p>
            </div>
          )}
        </section>

        {/* Previsão semanal + SHAP: fluxo principal */}

        <PrevisaoSemanal
          key={`semanal-${execucaoId}-${versaoHistorico}`}
          execucaoId={execucaoId}
          lojas={lojasDisponiveis}
          treinamentoConcluido={
            treinamentoSemanalConcluido
          }
        />

        {/* Previsão antiga preservada */}

        <PrevisaoFutura
          key={`${execucaoId}-${versaoHistorico}`}
          execucaoId={execucaoId}
          lojas={lojasDisponiveis}
          treinamentoConcluido={
            treinamentoDefinitivoConcluido
          }
        />

        {/* Atualização do histórico */}

       <AtualizacaoHistorico
  key={execucaoId}
  execucaoId={execucaoId}
  treinamentoConcluido={
    treinamentoDefinitivoConcluido ||
    treinamentoSemanalConcluido
  }
  onHistoricoAtualizado={() => {
    setVersaoHistorico(
      (versao) => versao + 1,
    )
  }}
/>

        <footer className="page-footer">
          Sistema de Predição de Vendas ·
          AutoML + XAI · Protótipo acadêmico

          {treinamentoSemanalConcluido &&
            ' · Modelo semanal disponível'}
        </footer>
      </main>
    </div>
  )
}

export default App
