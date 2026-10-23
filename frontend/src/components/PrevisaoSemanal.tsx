import { useRef, useState } from 'react'
import {
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
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, obterMensagemErro } from '../services/api'
type PrevisaoDia = {
  horizonte_dias: number
  data_prevista: string
  vendas_previstas: number
}
type ResultadoSemanal = {
  execucao_id: string
  loja_id: number
  data_origem: string
  algoritmo?: string
  tipo_modelo?: string
  horizonte_dias?: number
  previsoes: PrevisaoDia[]
  total_semanal_previsto: number
  observacao?: string
}
type ContribuicaoSHAP = {
  atributo: string
  contribuicao: number
}
type ExplicacaoSemanal = {
  execucao_id: string
  loja_id: number
  data_origem: string
  data_prevista: string
  horizonte_dias: number
  algoritmo: string
  vendas_previstas: number
  explicabilidade: {
    metodo: string
    valor_base: number
    previsao_bruta: number
    diferenca_reconstrucao: number
    principais_contribuicoes: ContribuicaoSHAP[]
    observacao: string
  }
}
type Props = {
  execucaoId: string
  lojas: string[]
  treinamentoConcluido: boolean
}
function formatarNumero(valor: number): string {
  return new Intl.NumberFormat('pt-BR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(valor)
}
function formatarData(data: string): string {
  const [ano, mes, dia] = data.slice(0, 10).split('-')
  return `${dia}/${mes}/${ano}`
}
function nomeAtributo(atributo: string): string {
  const nomes: Record<string, string> = {
    horizonte_dias: 'Distância até o dia previsto',
    lag_origem: 'Vendas no último dia registrado',
    lag_7_origem: 'Vendas há 7 dias',
    lag_14_origem: 'Vendas há 14 dias',
    lag_21_origem: 'Vendas há 21 dias',
    lag_28_origem: 'Vendas há 28 dias',
    media_7_origem: 'Média dos últimos 7 dias',
    media_28_origem: 'Média dos últimos 28 dias',
    observacoes_7_origem: 'Dias com vendas registradas (7 dias)',
    observacoes_28_origem: 'Dias com vendas registradas (28 dias)',
    dia_semana: 'Dia da semana',
    dia_mes: 'Dia do mês',
    mes: 'Mês',
    semana_ano: 'Semana do ano',
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
export default function PrevisaoSemanal({
execucaoId,
lojas,
treinamentoConcluido,
}: Props) {
  const [lojaId, setLojaId] = useState('')
  const [carregando, setCarregando] = useState(false)
  const [carregandoShap, setCarregandoShap] = useState(false)
  const [erro, setErro] = useState('')
  const [resultado, setResultado] =
    useState<ResultadoSemanal | null>(null)
  const [diaSelecionado, setDiaSelecionado] = useState(1)
  const [explicacao, setExplicacao] =
    useState<ExplicacaoSemanal | null>(null)
  const [mostrarShap, setMostrarShap] = useState(false)
  const requisicaoAtual = useRef(0)
  const lojaSelecionada =
    lojas.includes(lojaId) ? lojaId : lojas[0] || ''
  const habilitado =
    treinamentoConcluido &&
    Boolean(execucaoId) &&
    Boolean(lojaSelecionada)
  function limparResultados() {
    requisicaoAtual.current += 1
    setResultado(null)
    setExplicacao(null)
    setMostrarShap(false)
    setErro('')
    setCarregando(false)
    setCarregandoShap(false)
    setDiaSelecionado(1)
  }
  function selecionarDia(dia: number) {
    requisicaoAtual.current += 1
    setDiaSelecionado(dia)
    setExplicacao(null)
    setMostrarShap(false)
    setCarregandoShap(false)
    setErro('')
  }
  async function gerarPrevisao() {
    if (!habilitado) return
    const requisicao = ++requisicaoAtual.current
    setCarregando(true)
    setErro('')
    setResultado(null)
    setExplicacao(null)
    setMostrarShap(false)
    setDiaSelecionado(1)
    try {
      const resposta = await api.post<ResultadoSemanal>(
        `/previsoes/execucoes/${execucaoId}` +
          `/lojas/${lojaSelecionada}/semanal`,
      )
      if (requisicao !== requisicaoAtual.current) {
        return
      }
      const dados = resposta.data
      // A previsão registrada não inclui necessariamente metadados do modelo.
      // Eles são opcionais; os sete valores, as datas e a loja não são.
      const origem = typeof dados.data_origem === 'string'
        ? new Date(`${dados.data_origem}T00:00:00Z`)
        : new Date(Number.NaN)
      const previsoesValidas = Number.isFinite(origem.getTime()) &&
        Array.isArray(dados.previsoes) &&
        dados.previsoes.length === 7 &&
        dados.previsoes.every((item, indice) => {
          const dataEsperada = new Date(origem.getTime())
          dataEsperada.setUTCDate(dataEsperada.getUTCDate() + indice + 1)
          return item.horizonte_dias === indice + 1 &&
            item.data_prevista === dataEsperada.toISOString().slice(0, 10) &&
            Number.isFinite(item.vendas_previstas) &&
            item.vendas_previstas >= 0
        })
      if (
        (dados.execucao_id && dados.execucao_id !== execucaoId) ||
        String(dados.loja_id) !== lojaSelecionada ||
        !Number.isFinite(origem.getTime()) ||
        !previsoesValidas ||
        !Number.isFinite(dados.total_semanal_previsto) ||
        Math.abs(dados.total_semanal_previsto -
          dados.previsoes.reduce((soma, item) => soma + item.vendas_previstas, 0)) > 0.08
      ) {
        throw new Error(
          'A API retornou previsões semanais inconsistentes.',
        )
      }
      setResultado({ ...dados, execucao_id: execucaoId })
    } catch (erroCapturado) {
      if (requisicao !== requisicaoAtual.current) return
      setErro(obterMensagemErro(erroCapturado))
    } finally {
      if (requisicao === requisicaoAtual.current) {
        setCarregando(false)
      }
    }
  }
  async function explicarDia() {
    if (!resultado || carregandoShap) return
    if (
      explicacao &&
      explicacao.horizonte_dias === diaSelecionado
    ) {
      setMostrarShap(!mostrarShap)
      return
    }
    const requisicao = ++requisicaoAtual.current
    setCarregandoShap(true)
    setErro('')
    setExplicacao(null)
    try {
      const resposta = await api.post<ExplicacaoSemanal>(
        `/previsoes/execucoes/${execucaoId}` +
          `/lojas/${resultado.loja_id}` +
          `/semanal/explicacao/${diaSelecionado}`,
      )
      if (requisicao !== requisicaoAtual.current) return
      const dados = resposta.data
      const previsaoOriginal = resultado.previsoes.find(
        (item) => item.horizonte_dias === diaSelecionado,
      )
      if (
        !previsaoOriginal ||
        dados.execucao_id !== resultado.execucao_id ||
        dados.loja_id !== resultado.loja_id ||
        dados.data_origem !== resultado.data_origem ||
        dados.data_prevista !== previsaoOriginal.data_prevista ||
        dados.horizonte_dias !== diaSelecionado ||
        (resultado.algoritmo && dados.algoritmo !== resultado.algoritmo) ||
        Math.abs(
          dados.vendas_previstas -
            previsaoOriginal.vendas_previstas,
        ) > 0.02
      ) {
        throw new Error(
          'O histórico mudou desde a previsão. ' +
            'Gere uma nova previsão semanal antes de explicar.',
        )
      }
      setExplicacao(dados)
      setMostrarShap(true)
    } catch (erroCapturado) {
      if (requisicao !== requisicaoAtual.current) return
      setErro(obterMensagemErro(erroCapturado))
    } finally {
      if (requisicao === requisicaoAtual.current) {
        setCarregandoShap(false)
      }
    }
  }
  const dadosGrafico =
    resultado?.previsoes.map((item) => ({
      dia: `Dia ${item.horizonte_dias}`,
      data: formatarData(item.data_prevista),
      vendas: item.vendas_previstas,
    })) ?? []
  const contribuicoes =
    explicacao?.explicabilidade.principais_contribuicoes.map(
      (item) => ({
        atributo: nomeAtributo(item.atributo),
        nomeOriginal: item.atributo,
        contribuicao: item.contribuicao,
      }),
    ) ?? []
  return (
    <section
className="panel previsao-panel"
id="previsao-semanal"
    >
      <div className="panel-heading">
        <div>
          <h2>Previsão dos próximos 7 dias</h2>
          <p>
            Estimativas diárias, total semanal e
            explicações SHAP por dia.
          </p>
        </div>
        <CalendarDays className="heading-icon" size={23} />
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
                <option value="">
                  Nenhuma loja disponível
                </option>
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
disabled={!habilitado || carregando}
onClick={() => void gerarPrevisao()}
        >
          {carregando ? (
            <LoaderCircle className="spin" size={18} />
          ) : (
            <Sparkles size={18} />
          )}
          {carregando
            ? 'Calculando semana...'
            : 'Gerar previsão semanal'}
        </button>
      </div>
      {!treinamentoConcluido && (
        <p className="previsao-ajuda">
          Selecione uma execução com treinamento
          definitivo concluído.
        </p>
      )}
      {erro && (
        <div className="previsao-erro" role="alert">
          <TriangleAlert size={18} />
          <span>{erro}</span>
        </div>
      )}
      {resultado && (
        <div className="previsao-resultado">
          <div className="previsao-resultado-topo">
            <div className="previsao-sucesso">
              <CheckCircle2 size={18} />
              Previsão semanal calculada
            </div>
            <span className="previsao-algoritmo">
              {resultado.algoritmo?.toUpperCase() ?? 'AUTOML'}
            </span>
          </div>
          <div className="previsao-valor">
            <span>Total estimado dos sete dias</span>
            <strong>
              {formatarNumero(
                resultado.total_semanal_previsto,
              )}
            </strong>
            <small>
              Loja {resultado.loja_id} · última venda em{' '}
              {formatarData(resultado.data_origem)}
            </small>
          </div>
          <h3>Estimativa por dia</h3>
          <div
style={{
              width: '100%',
              height: 290,
              marginTop: 16,
              marginBottom: 22,
            }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
data={dadosGrafico}
margin={{
                  top: 12,
                  right: 18,
                  bottom: 5,
                  left: 0,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="dia" />
                <YAxis width={75} />
                <Tooltip
formatter={(valor) =>
                    formatarNumero(Number(valor))
                  }
                />
                <Line
name="Vendas previstas"
dataKey="vendas"
type="monotone"
stroke="#21836a"
strokeWidth={3}
dot={{ r: 4 }}
activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table
style={{
                width: '100%',
                borderCollapse: 'collapse',
                textAlign: 'left',
              }}
            >
              <thead>
                <tr>
                  <th scope="col" style={{ padding: 12 }}>
                    Dia
                  </th>
                  <th scope="col" style={{ padding: 12 }}>
                    Data
                  </th>
                  <th scope="col" style={{ padding: 12 }}>
                    Vendas previstas
                  </th>
                </tr>
              </thead>
              <tbody>
                {resultado.previsoes.map((item) => (
                  <tr key={item.horizonte_dias}>
                    <td
style={{
                        padding: 12,
                        borderTop:
                          '1px solid rgba(128,128,128,.2)',
                      }}
                    >
                      Dia {item.horizonte_dias}
                    </td>
                    <td
style={{
                        padding: 12,
                        borderTop:
                          '1px solid rgba(128,128,128,.2)',
                      }}
                    >
                      {formatarData(item.data_prevista)}
                    </td>
                    <td
style={{
                        padding: 12,
                        borderTop:
                          '1px solid rgba(128,128,128,.2)',
                      }}
                    >
                      {formatarNumero(
                        item.vendas_previstas,
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr>
                  <th
scope="row"
colSpan={2}
style={{ padding: 12 }}
                  >
                    Total semanal
                  </th>
                  <th style={{ padding: 12 }}>
                    {formatarNumero(
                      resultado.total_semanal_previsto,
                    )}
                  </th>
                </tr>
              </tfoot>
            </table>
          </div>
          <div
className="shap-container"
style={{ marginTop: 24 }}
          >
            <div className="shap-cabecalho">
              <div>
                <h3>Entenda uma previsão diária</h3>
                <p>
                  Escolha um dos sete dias para visualizar
                  os fatores utilizados pelo modelo.
                </p>
              </div>
              <span className="shap-metodo">SHAP</span>
            </div>
            <div
className="previsao-controles"
style={{ marginTop: 16 }}
            >
              <label className="previsao-campo">
                <span>Dia a explicar</span>
                <div className="previsao-select">
                  <CalendarDays size={18} />
                  <select
value={diaSelecionado}
disabled={carregandoShap}
onChange={(evento) =>
                      selecionarDia(Number(evento.target.value))
                    }
                  >
                    {resultado.previsoes.map((item) => (
                      <option
key={item.horizonte_dias}
value={item.horizonte_dias}
                      >
                        Dia {item.horizonte_dias} —{' '}
                        {formatarData(item.data_prevista)}
                      </option>
                    ))}
                  </select>
                </div>
              </label>
              <button
type="button"
className="secondary-button"
disabled={carregandoShap}
onClick={() => void explicarDia()}
              >
                {carregandoShap ? (
                  <LoaderCircle
size={18}
className="spin"
                  />
                ) : (
                  <BarChart3 size={18} />
                )}
                {carregandoShap
                  ? 'Calculando SHAP...'
                  : explicacao && mostrarShap
                    ? 'Ocultar explicação'
                    : explicacao
                      ? 'Mostrar explicação'
                      : 'Explicar previsão do dia'}
                {explicacao &&
                  (mostrarShap ? (
                    <ChevronUp size={17} />
                  ) : (
                    <ChevronDown size={17} />
                  ))}
              </button>
            </div>
            {explicacao && mostrarShap && (
              <div style={{ marginTop: 20 }}>
                <div className="shap-resumo">
                  <div>
                    <span>Referência do modelo</span>
                    <strong>
                      {formatarNumero(
                        explicacao.explicabilidade.valor_base,
                      )}
                    </strong>
                  </div>
                  <div>
                    <span>
                      Estimativa de {formatarData(
                        explicacao.data_prevista,
                      )}
                    </span>
                    <strong>
                      {formatarNumero(
                        explicacao.vendas_previstas,
                      )}
                    </strong>
                  </div>
                </div>
                <p className="previsao-observacao">
                  As barras verdes mostram atributos que
                  aumentaram a estimativa em relação ao
                  valor de referência. As barras laranjas
                  mostram atributos que a reduziram.
                </p>
                <div
className="shap-grafico"
style={{
                    height: Math.max(
                      290,
                      contribuicoes.length * 44 + 50,
                    ),
                    marginTop: 16,
                  }}
                >
                  <ResponsiveContainer
width="100%"
height="100%"
                  >
                    <BarChart
data={contribuicoes}
layout="vertical"
margin={{
                        top: 10,
                        right: 28,
                        bottom: 8,
                        left: 12,
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
width={205}
tick={{ fontSize: 11 }}
                      />
                      <Tooltip
formatter={(valor) => [
                          formatarNumero(Number(valor)),
                          'Contribuição SHAP',
                        ]}
                      />
                      <ReferenceLine
x={0}
stroke="#7e8b98"
                      />
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
                  {explicacao.explicabilidade.observacao}
                </p>
              </div>
            )}
          </div>
          <p className="previsao-observacao">
            {resultado.observacao ?? 'Previsões operacionais registradas para esta semana.'}
          </p>
        </div>
      )}
    </section>
  )
}
