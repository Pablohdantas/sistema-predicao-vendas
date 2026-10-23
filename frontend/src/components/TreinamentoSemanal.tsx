
import { useEffect, useState } from 'react'

import {
  BrainCircuit,
  CheckCircle2,
  LoaderCircle,
  TriangleAlert,
} from 'lucide-react'

import { api, obterMensagemErro } from '../services/api'

type StatusSemanal = {
  execucao_id: string
  status:
    | 'nao_iniciado'
    | 'em_andamento'
    | 'concluido'
    | 'falhou'
    | 'interrompido'
  tipo: string
  algoritmo_vencedor?: string
  otimizacao_flaml?: boolean
  erro?: string | null
}

type Props = {
  execucaoId: string
  onModeloDisponivel: (disponivel: boolean) => void
}

type Perfil = 'rapido' | 'ampliado' | 'personalizado'

type Configuracao = {
  tentativas: number
  tempo_por_algoritmo: number
}

const PERFIS: Record<
  Exclude<Perfil, 'personalizado'>,
  Configuracao
> = {
  rapido: {
    tentativas: 3,
    tempo_por_algoritmo: 30,
  },
  ampliado: {
    tentativas: 10,
    tempo_por_algoritmo: 180,
  },
}

export default function TreinamentoSemanal({
  execucaoId,
  onModeloDisponivel,
}: Props) {
  const [status, setStatus] =
    useState<StatusSemanal | null>(null)

  const [perfil, setPerfil] = useState<Perfil>('rapido')

  const [configuracao, setConfiguracao] =
    useState<Configuracao>(PERFIS.rapido)

  const [iniciando, setIniciando] = useState(false)
  const [erro, setErro] = useState('')

  const emAndamento =
    status?.status === 'em_andamento'

  const concluido =
    status?.status === 'concluido'

  useEffect(() => {
    if (!execucaoId) return

    let ativo = true

    const consultar = async () => {
      try {
        const resposta = await api.get<StatusSemanal>(
          `/treinamentos-semanais/${execucaoId}`,
        )

        if (!ativo) return

        setStatus(resposta.data)
        setErro('')

        onModeloDisponivel(
          resposta.data.status === 'concluido',
        )
      } catch (erroCapturado) {
        if (!ativo) return
        setErro(obterMensagemErro(erroCapturado))
        onModeloDisponivel(false)
      }
    }

    void consultar()

    const intervalo = window.setInterval(() => {
      void consultar()
    }, 3000)

    return () => {
      ativo = false
      window.clearInterval(intervalo)
    }
  }, [execucaoId, onModeloDisponivel])

  function selecionarPerfil(novoPerfil: Perfil) {
    setPerfil(novoPerfil)

    if (novoPerfil !== 'personalizado') {
      setConfiguracao(PERFIS[novoPerfil])
    }
  }

  function alterarParametro(
    campo: keyof Configuracao,
    valor: number,
  ) {
    setPerfil('personalizado')

    setConfiguracao((anterior) => ({
      ...anterior,
      [campo]: valor,
    }))
  }

  const parametrosValidos =
    Number.isInteger(configuracao.tentativas) &&
    configuracao.tentativas >= 1 &&
    configuracao.tentativas <= 30 &&
    Number.isInteger(configuracao.tempo_por_algoritmo) &&
    configuracao.tempo_por_algoritmo >= 1 &&
    configuracao.tempo_por_algoritmo <= 1800

  async function iniciar() {
    if (
      !execucaoId ||
      iniciando ||
      emAndamento ||
      concluido ||
      !parametrosValidos
    ) {
      return
    }

    setIniciando(true)
    setErro('')

    try {
      const resposta = await api.post<StatusSemanal>(
        `/treinamentos-semanais/${execucaoId}`,
        configuracao,
      )

      setStatus(resposta.data)
    } catch (erroCapturado) {
      setErro(obterMensagemErro(erroCapturado))
    } finally {
      setIniciando(false)
    }
  }

  const bloqueado =
    iniciando || emAndamento || concluido

  return (
    <section className="panel" id="treinamento-semanal">
      <div className="panel-heading">
        <div>
          <h2>Treinamento AutoML semanal</h2>
          <p>
            O sistema compara LightGBM e XGBoost
            para prever os próximos sete dias.
          </p>
        </div>

        <BrainCircuit className="heading-icon" size={23} />
      </div>

      {concluido ? (
        <div className="previsao-sucesso">
          <CheckCircle2 size={18} />
          <span>
            Modelo semanal disponível
            {status?.algoritmo_vencedor
              ? ` · ${status.algoritmo_vencedor.toUpperCase()}`
              : ''}
          </span>
        </div>
      ) : (
        <>
          <div
            style={{
              marginTop: 14,
              marginBottom: 16,
              display: 'grid',
              gap: 12,
            }}
          >
            <label className="previsao-campo">
              <span>Perfil de treinamento</span>

              <select
                className="previsao-select"
                value={perfil}
                disabled={bloqueado}
                onChange={(evento) =>
                  selecionarPerfil(
                    evento.target.value as Perfil,
                  )
                }
                style={{
                  padding: 12,
                  width: '100%',
                }}
              >
                <option value="rapido">
                  Rápido — 3 tentativas, 30 segundos
                </option>
                <option value="ampliado">
                  Ampliado — 10 tentativas, 180 segundos
                </option>
                <option value="personalizado">
                  Personalizado
                </option>
              </select>
            </label>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns:
                  'repeat(auto-fit, minmax(180px, 1fr))',
                gap: 12,
              }}
            >
              <label className="previsao-campo">
                <span>Tentativas por algoritmo (1 a 30)</span>
                <input
                  type="number"
                  min={1}
                  max={30}
                  step={1}
                  value={configuracao.tentativas}
                  disabled={bloqueado}
                  onChange={(evento) =>
                    alterarParametro(
                      'tentativas',
                      Number(evento.target.value),
                    )
                  }
                  style={{
                    width: '100%',
                    padding: 12,
                  }}
                />
              </label>

              <label className="previsao-campo">
                <span>Tempo por algoritmo, em segundos</span>
                <input
                  type="number"
                  min={1}
                  max={1800}
                  step={1}
                  value={configuracao.tempo_por_algoritmo}
                  disabled={bloqueado}
                  onChange={(evento) =>
                    alterarParametro(
                      'tempo_por_algoritmo',
                      Number(evento.target.value),
                    )
                  }
                  style={{
                    width: '100%',
                    padding: 12,
                  }}
                />
              </label>
            </div>
          </div>

          <p className="previsao-ajuda">
            O modelo será selecionado pelo RMSE em
            validação temporal e avaliado em um
            período externo reservado. O limite atual
            é de 5.000 registros por dataset.
          </p>

          <button
            type="button"
            className="primary-button"
            disabled={
              !execucaoId ||
              bloqueado ||
              !parametrosValidos
            }
            onClick={() => void iniciar()}
          >
            {iniciando || emAndamento ? (
              <LoaderCircle
                className="spin"
                size={18}
              />
            ) : (
              <BrainCircuit size={18} />
            )}

            {iniciando || emAndamento
              ? 'Treinando modelo semanal...'
              : 'Iniciar AutoML semanal'}
          </button>
        </>
      )}

      {status?.status === 'falhou' && (
        <div className="previsao-erro" role="alert">
          <TriangleAlert size={18} />
          <span>
            {status.erro || 'O treinamento falhou.'}
          </span>
        </div>
      )}

      {status?.status === 'interrompido' && (
        <div className="previsao-erro" role="alert">
          <TriangleAlert size={18} />
          <span>
            O treinamento foi interrompido.
          </span>
        </div>
      )}

      {erro && (
        <div className="previsao-erro" role="alert">
          <TriangleAlert size={18} />
          <span>{erro}</span>
        </div>
      )}
    </section>
  )
}
