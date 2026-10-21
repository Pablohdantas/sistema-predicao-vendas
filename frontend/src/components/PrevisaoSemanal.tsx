
import { useRef, useState } from 'react'

import {
  CalendarDays,
  CheckCircle2,
  LoaderCircle,
  Sparkles,
  Store,
  TriangleAlert,
} from 'lucide-react'

import {
  CartesianGrid,
  Line,
  LineChart,
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
  algoritmo: string
  tipo_modelo: string
  horizonte_dias: number
  previsoes: PrevisaoDia[]
  total_semanal_previsto: number
  observacao: string
}

type Props = {
  execucaoId: string
  lojas: string[]
  treinamentoConcluido: boolean
}

const formatarNumero = (valor: number) =>
  new Intl.NumberFormat('pt-BR', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(valor)

function formatarData(data: string) {
  const [ano, mes, dia] = data.slice(0, 10).split('-')
  return `${dia}/${mes}/${ano}`
}

export default function PrevisaoSemanal({
  execucaoId,
  lojas,
  treinamentoConcluido,
}: Props) {
  const [lojaId, setLojaId] = useState('')
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState('')
  const [resultado, setResultado] =
    useState<ResultadoSemanal | null>(null)

  const requisicaoAtual = useRef(0)

  const lojaSelecionada =
    lojas.includes(lojaId) ? lojaId : lojas[0] || ''

  const habilitado =
    treinamentoConcluido &&
    !!execucaoId &&
    !!lojaSelecionada

  function limparResultados() {
    requisicaoAtual.current += 1
    setResultado(null)
    setErro('')
    setCarregando(false)
  }

  async function gerarPrevisao() {
    if (!habilitado) return

    const requisicao = ++requisicaoAtual.current

    setCarregando(true)
    setErro('')
    setResultado(null)

    try {
      const resposta = await api.post<ResultadoSemanal>(
        `/previsoes/execucoes/${execucaoId}` +
          `/lojas/${lojaSelecionada}/semanal`,
      )

      if (requisicao !== requisicaoAtual.current) {
        return
      }

      const dados = resposta.data

      if (
        dados.execucao_id !== execucaoId ||
        String(dados.loja_id) !== lojaSelecionada ||
        dados.previsoes.length !== 7 ||
        dados.previsoes.some(
          (item, indice) =>
            item.horizonte_dias !== indice + 1 ||
            !Number.isFinite(item.vendas_previstas),
        )
      ) {
        throw new Error(
          'O backend retornou uma previsão semanal inconsistente.',
        )
      }

      setResultado(dados)
    } catch (erroCapturado) {
      if (requisicao !== requisicaoAtual.current) {
        return
      }

      setErro(obterMensagemErro(erroCapturado))
    } finally {
      if (requisicao === requisicaoAtual.current) {
        setCarregando(false)
      }
    }
  }

  const dadosGrafico =
    resultado?.previsoes.map((item) => ({
      dia: `Dia ${item.horizonte_dias}`,
      vendas: item.vendas_previstas,
      data: formatarData(item.data_prevista),
    })) ?? []

  return (
    <section className="panel previsao-panel" id="previsao-semanal">
      <div className="panel-heading">
        <div>
          <h2>Previsão dos próximos 7 dias</h2>
          <p>
            Estimativas diárias e total semanal para uma loja.
          </p>
        </div>

        <CalendarDays className="heading-icon" size={23} />
      </div>

      <p className="previsao-ajuda">
        Funcionalidade experimental: utiliza um modelo
        multihorizonte treinado separadamente.
      </p>

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
                <option key={loja} value={loja}>
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
              {resultado.algoritmo.toUpperCase()}
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
                  labelFormatter={(rotulo, itens) => {
                    const primeiro = itens?.[0]?.payload
                    return primeiro?.data
                      ? `${rotulo} · ${primeiro.data}`
                      : String(rotulo)
                  }}
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

          <p className="previsao-observacao">
            {resultado.observacao}
          </p>
        </div>
      )}
    </section>
  )
}
