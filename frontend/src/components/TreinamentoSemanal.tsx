
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
  erro?: string | null
}

type Props = {
  execucaoId: string
  treinamentoConcluido: boolean
}

export default function TreinamentoSemanal({
  execucaoId,
  treinamentoConcluido,
}: Props) {
  const [status, setStatus] =
    useState<StatusSemanal | null>(null)

  const [iniciando, setIniciando] =
    useState(false)

  const [erro, setErro] =
    useState('')

  useEffect(() => {
    if (!execucaoId) {
      return
    }

    let ativo = true

    async function consultar() {
      try {
        const resposta = await api.get<StatusSemanal>(
          `/treinamentos-semanais/${execucaoId}`,
        )

        if (ativo) {
          setStatus(resposta.data)
        }

      } catch (erroCapturado) {
        if (ativo) {
          setErro(
            obterMensagemErro(erroCapturado),
          )
        }
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
  }, [execucaoId])

  async function iniciar() {
    if (!execucaoId || iniciando) {
      return
    }

    setIniciando(true)
    setErro('')

    try {
      const resposta = await api.post<StatusSemanal>(
        `/treinamentos-semanais/${execucaoId}`,
      )

      setStatus(resposta.data)

    } catch (erroCapturado) {
      setErro(
        obterMensagemErro(erroCapturado),
      )

    } finally {
      setIniciando(false)
    }
  }

  const treinando =
    iniciando ||
    status?.status === 'em_andamento'

  const concluido =
    status?.status === 'concluido'

  const podeTreinar =
    Boolean(execucaoId) &&
    treinamentoConcluido &&
    !treinando &&
    !concluido

  return (
    <section
      className="panel"
      id="treinamento-semanal"
    >
      <div className="panel-heading">
        <div>
          <h2>Treinamento semanal</h2>

          <p>
            Treine um modelo experimental
            para prever os próximos sete dias.
          </p>
        </div>

        <BrainCircuit
          className="heading-icon"
          size={23}
        />
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
          <p className="previsao-ajuda">
            Comparação experimental entre
            LightGBM e XGBoost.

            Esta versão utiliza configurações
            fixas e aceita datasets de
            até 5.000 registros.
          </p>

          <button
            type="button"
            className="primary-button"
            disabled={!podeTreinar}
            onClick={() => void iniciar()}
          >
            {treinando ? (
              <LoaderCircle
                className="spin"
                size={18}
              />
            ) : (
              <BrainCircuit size={18} />
            )}

            {treinando
              ? 'Treinando modelo semanal...'
              : 'Treinar modelo semanal'}
          </button>
        </>
      )}

      {status?.status === 'falhou' && (
        <div
          className="previsao-erro"
          role="alert"
        >
          <TriangleAlert size={18} />

          <span>
            {status.erro ||
              'O treinamento semanal falhou.'}
          </span>
        </div>
      )}

      {status?.status === 'interrompido' && (
        <div
          className="previsao-erro"
          role="alert"
        >
          <TriangleAlert size={18} />

          <span>
            O treinamento semanal foi interrompido.
          </span>
        </div>
      )}

      {!treinamentoConcluido && (
        <p className="previsao-ajuda">
          Conclua primeiro o treinamento
          definitivo convencional para habilitar
          o treinamento semanal.
        </p>
      )}

      {erro && (
        <div
          className="previsao-erro"
          role="alert"
        >
          <TriangleAlert size={18} />
          <span>{erro}</span>
        </div>
      )}
    </section>
  )
}
