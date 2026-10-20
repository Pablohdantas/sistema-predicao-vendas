
import { useRef, useState } from 'react'
import {
  ArrowRight,
  BarChart3,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  LoaderCircle,
  Sparkles,
  Store,
  TriangleAlert,
} from 'lucide-react'

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { api, obterMensagemErro } from '../services/api'

type PrevisaoAutomatica = {
  execucao_id: string
  loja_id: number
  data_origem: string
  data_prevista: string
  horizonte_dias: number
  vendas_previstas: number
  algoritmo: string
  modo: string
  variaveis_futuras_desconhecidas: string[]
  observacao: string
}

type ContribuicaoSHAP = {
  atributo: string
  contribuicao: number
}

type ExplicacaoAutomatica = {
  execucao_id: string
  loja_id: number
  data_origem: string
  data_prevista: string
  algoritmo: string
  vendas_previstas: number
  explicabilidade: {
    metodo: string
    valor_base: number
    principais_contribuicoes: ContribuicaoSHAP[]
    diferenca_reconstrucao: number
    observacao: string
  }
}

type Props = {
  execucaoId: string
  lojas: string[]
  treinamentoConcluido: boolean
}

function formatarData(data: string): string {
  const [ano, mes, dia] = data.split('-')
  return `${dia}/${mes}/${ano}`
}

function formatarNumero(numero: number): string {
  return new Intl.NumberFormat('pt-BR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(numero)
}

function nomeAtributo(atributo: string): string {
  const nomes: Record<string, string> = {
    lag_7: 'Vendas há 7 dias',
    lag_14: 'Vendas há 14 dias',
    lag_21: 'Vendas há 21 dias',
    lag_28: 'Vendas há 28 dias',
    media_movel_7: 'Média móvel de 7 dias',
    media_movel_28: 'Média móvel de 28 dias',
    observacoes_media_7: 'Observações em 7 dias',
    observacoes_media_28: 'Observações em 28 dias',
    dia_semana: 'Dia da semana',
    dia_mes: 'Dia do mês',
    mes: 'Mês',
    semana_ano: 'Semana do ano',
    promocao: 'Promoção',
    promocao_continua: 'Promoção contínua',
    mes_renovacao_promocao: 'Renovação da promoção',
    feriado: 'Feriado',
    feriado_escolar: 'Feriado escolar',
    distancia_concorrente: 'Distância do concorrente',
  }

  if (atributo.startsWith('tipo_loja_')) {
    return `Tipo de loja: ${atributo.slice('tipo_loja_'.length)}`
  }

  if (atributo.startsWith('sortimento_')) {
    return `Sortimento: ${atributo.slice('sortimento_'.length)}`
  }

  return nomes[atributo] ?? atributo.replaceAll('_', ' ')
}

export default function PrevisaoFutura({
  execucaoId,
  lojas,
  treinamentoConcluido,
}: Props) {
  const [lojaId, setLojaId] = useState('')
  const [carregando, setCarregando] = useState(false)
  const [carregandoShap, setCarregandoShap] = useState(false)
  const [erro, setErro] = useState('')
  const [resultado, setResultado] =
    useState<PrevisaoAutomatica | null>(null)
  const [explicacao, setExplicacao] =
    useState<ExplicacaoAutomatica | null>(null)
  const [mostrarExplicacao, setMostrarExplicacao] = useState(false)

  // Impede que uma resposta antiga substitua outra previsão.
  const requisicaoAtual = useRef(0)

  const lojaSelecionada = lojaId || lojas[0] || ''

  function limparResultados() {
    requisicaoAtual.current += 1
    setResultado(null)
    setExplicacao(null)
    setMostrarExplicacao(false)
    setErro('')
    setCarregando(false)
    setCarregandoShap(false)
  }

  async function gerarPrevisao() {
    if (!execucaoId || !lojaSelecionada) {
      setErro('Selecione uma execução e uma loja.')
      return
    }

    const requisicao = ++requisicaoAtual.current

    setCarregando(true)
    setErro('')
    setResultado(null)
    setExplicacao(null)
    setMostrarExplicacao(false)

    try {
      const resposta = await api.post<PrevisaoAutomatica>(
        `/previsoes/execucoes/${execucaoId}` +
          `/lojas/${lojaSelecionada}/automatica`,
      )

      if (requisicao !== requisicaoAtual.current) return

      setResultado(resposta.data)
    } catch (erroCapturado) {
      if (requisicao !== requisicaoAtual.current) return
      setErro(obterMensagemErro(erroCapturado))
    } finally {
      if (requisicao === requisicaoAtual.current) {
        setCarregando(false)
      }
    }
  }

  async function explicarResultado() {
    if (!resultado) return

    if (explicacao) {
      setMostrarExplicacao(!mostrarExplicacao)
      return
    }

    const requisicao = requisicaoAtual.current

    setCarregandoShap(true)
    setErro('')

    try {
      const resposta = await api.post<ExplicacaoAutomatica>(
        `/previsoes/execucoes/${execucaoId}` +
          `/lojas/${resultado.loja_id}/automatica/explicacao`,
      )

      if (requisicao !== requisicaoAtual.current) return

      const dados = resposta.data

      if (
        dados.execucao_id !== resultado.execucao_id ||
        dados.loja_id !== resultado.loja_id ||
        dados.data_origem !== resultado.data_origem ||
        dados.data_prevista !== resultado.data_prevista ||
        dados.algoritmo !== resultado.algoritmo ||
        Math.abs(
          dados.vendas_previstas - resultado.vendas_previstas,
        ) > 0.02
      ) {
        throw new Error(
          'O histórico mudou desde a previsão. ' +
          'Gere uma nova previsão antes de solicitar o SHAP.',
        )
      }

      setExplicacao(dados)
      setMostrarExplicacao(true)
    } catch (erroCapturado) {
      if (requisicao !== requisicaoAtual.current) return
      setErro(obterMensagemErro(erroCapturado))
    } finally {
      if (requisicao === requisicaoAtual.current) {
        setCarregandoShap(false)
      }
    }
  }

  const contribuicoes = (
    explicacao?.explicabilidade.principais_contribuicoes ?? []
  ).map((item) => ({
    atributo: nomeAtributo(item.atributo),
    nomeOriginal: item.atributo,
    contribuicao: item.contribuicao,
  }))

  return (
    <section className="panel previsao-panel" id="previsao-futura">
      <div className="panel-heading">
        <div>
          <h2>Previsões futuras</h2>
          <p>
            Estime as vendas e descubra quais atributos
            influenciaram o modelo.
          </p>
        </div>
        <Sparkles className="heading-icon" size={23} />
      </div>

      <div className="previsao-controles">
        <label className="previsao-campo">
          <span>Loja</span>
          <div className="previsao-select">
            <Store size={18} />
            <select
              value={lojaSelecionada}
              disabled={!treinamentoConcluido || carregando}
              onChange={(evento) => {
                setLojaId(evento.target.value)
                limparResultados()
              }}
            >
              {lojas.length === 0 && (
                <option value="">Nenhuma loja disponível</option>
              )}
              {lojas.map((loja) => (
                <option value={loja} key={loja}>
                  Loja {loja}
                </option>
              ))}
            </select>
          </div>
        </label>

        <button
          type="button"
          className="primary-button previsao-botao"
          disabled={
            !treinamentoConcluido ||
            !lojaSelecionada ||
            carregando
          }
          onClick={() => void gerarPrevisao()}
        >
          {carregando ? (
            <LoaderCircle className="spin" size={18} />
          ) : (
            <Sparkles size={18} />
          )}
          {carregando ? 'Calculando...' : 'Gerar previsão'}
        </button>
      </div>

      {!treinamentoConcluido && (
        <p className="previsao-ajuda">
          Conclua um treinamento definitivo para habilitar
          as previsões futuras.
        </p>
      )}

      {treinamentoConcluido && lojas.length === 0 && (
        <p className="previsao-ajuda">
          Nenhuma loja foi identificada nos resultados disponíveis.
        </p>
      )}

      {erro && (
        <div className="previsao-erro" role="alert">
          <TriangleAlert size={18} />
          {erro}
        </div>
      )}

      {resultado && (
        <div className="previsao-resultado">
          <div className="previsao-resultado-topo">
            <div className="previsao-sucesso">
              <CheckCircle2 size={18} />
              Previsão calculada
            </div>
            <span className="previsao-algoritmo">
              {resultado.algoritmo.toUpperCase()}
            </span>
          </div>

          <div className="previsao-valor">
            <span>Vendas estimadas</span>
            <strong>
              {formatarNumero(resultado.vendas_previstas)}
            </strong>
            <small>
              Para a loja {resultado.loja_id}, em{' '}
              {formatarData(resultado.data_prevista)}
            </small>
          </div>

          <div className="previsao-periodo">
            <div>
              <CalendarDays size={18} />
              <span>Último registro</span>
              <strong>
                {formatarData(resultado.data_origem)}
              </strong>
            </div>
            <ArrowRight size={20} className="previsao-seta" />
            <div>
              <CalendarDays size={18} />
              <span>Data prevista</span>
              <strong>
                {formatarData(resultado.data_prevista)}
              </strong>
            </div>
          </div>

          {resultado.variaveis_futuras_desconhecidas.length > 0 && (
            <p className="previsao-aviso">
              <TriangleAlert size={17} />
              Informações futuras desconhecidas:{' '}
              {resultado.variaveis_futuras_desconhecidas.join(', ')}.
              O modelo recebeu esses valores como ausentes.
            </p>
          )}

          <p className="previsao-observacao">
            Esta é uma estimativa para uma data específica,
            sete dias após a última observação. Não representa
            a soma das vendas da semana.
          </p>

          <div className="shap-acoes">
            <button
              type="button"
              className="secondary-button shap-botao"
              disabled={carregandoShap}
              onClick={() => void explicarResultado()}
            >
              {carregandoShap ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <BarChart3 size={18} />
              )}

              {carregandoShap
                ? 'Calculando SHAP...'
                : explicacao && mostrarExplicacao
                  ? 'Ocultar explicação'
                  : explicacao
                    ? 'Mostrar explicação'
                    : 'Explicar previsão'}

              {explicacao && (
                mostrarExplicacao
                  ? <ChevronUp size={17} />
                  : <ChevronDown size={17} />
              )}
            </button>
          </div>

          {explicacao && mostrarExplicacao && (
            <div className="shap-container">
              <div className="shap-cabecalho">
                <div>
                  <h3>Por que o modelo fez esta previsão?</h3>
                  <p>
                    Contribuições SHAP dos atributos mais
                    relevantes para esta estimativa.
                  </p>
                </div>
                <span className="shap-metodo">
                  {explicacao.explicabilidade.metodo}
                </span>
              </div>

              <div className="shap-resumo">
                <div>
                  <span>Valor base do modelo</span>
                  <strong>
                    {formatarNumero(
                      explicacao.explicabilidade.valor_base,
                    )}
                  </strong>
                </div>
                <div>
                  <span>Previsão explicada</span>
                  <strong>
                    {formatarNumero(explicacao.vendas_previstas)}
                  </strong>
                </div>
              </div>

              <div className="shap-legenda">
                <span>
                  <i className="shap-legenda-positivo" />
                  Aumenta a estimativa
                </span>
                <span>
                  <i className="shap-legenda-negativo" />
                  Reduz a estimativa
                </span>
              </div>

              <div
                className="shap-grafico"
                style={{
                  height: Math.max(
                    270,
                    contribuicoes.length * 43 + 40,
                  ),
                }}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={contribuicoes}
                    layout="vertical"
                    margin={{
                      top: 10,
                      right: 25,
                      left: 12,
                      bottom: 5,
                    }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      horizontal={false}
                    />
                    <XAxis
                      type="number"
                      tickFormatter={(valor: number) =>
                        formatarNumero(valor)
                      }
                    />
                    <YAxis
                      dataKey="atributo"
                      type="category"
                      width={170}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      formatter={(valor) => [
                        formatarNumero(Number(valor)),
                        'Contribuição SHAP',
                      ]}
                    />
                    <ReferenceLine x={0} stroke="#7e8b98" />
                    <Bar
                      dataKey="contribuicao"
                      radius={[3, 3, 3, 3]}
                      maxBarSize={23}
                    >
                      {contribuicoes.map((item) => (
                        <Cell
                          key={item.nomeOriginal}
                          fill={
                            item.contribuicao >= 0
                              ? '#21836a'
                              : '#d98246'
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <p className="shap-nota">
                As contribuições representam o efeito de cada
                atributo sobre a saída do modelo em relação ao
                valor base. O gráfico mostra apenas os atributos
                principais, não necessariamente todos.
              </p>

              <p className="shap-nota">
                {explicacao.explicabilidade.observacao}
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
