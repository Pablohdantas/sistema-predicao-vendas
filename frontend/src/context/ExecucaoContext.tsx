import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
import type { ReactNode } from 'react'
import { api, obterMensagemErro } from '../services/api'
import {
  CHAVE_EXECUCAO,
  ExecucaoContext,
  type StatusSemanal,
} from './execucaoContext'

type RespostaExecucoes = {
  execucoes: { execucao_id: string }[]
}

export function ExecucaoProvider({ children }: { children: ReactNode }) {
  const [idInicial] = useState(() => localStorage.getItem(CHAVE_EXECUCAO) ?? '')
  const [execucaoId, setExecucaoId] = useState('')
  const [statusSemanal, setStatusSemanal] = useState<StatusSemanal | null>(null)
  const [carregando, setCarregando] = useState(Boolean(idInicial))
  const [erro, setErro] = useState('')
  const atualRef = useRef('')
  const solicitacaoRef = useRef(0)

  const consultarStatus = useCallback(async (id: string) => {
    const resposta = await api.get<StatusSemanal>(`/treinamentos-semanais/${id}`)
    if (atualRef.current === id) {
      setStatusSemanal(resposta.data)
    }
  }, [])

  const selecionarExecucao = useCallback(async (id: string): Promise<boolean> => {
    if (!id) return false
    const solicitacao = ++solicitacaoRef.current
    atualRef.current = ''
    setCarregando(true)
    setErro('')
    setStatusSemanal(null)

    try {
      const resposta = await api.get<RespostaExecucoes>('/execucoes/')
      if (solicitacao !== solicitacaoRef.current) return false
      if (!resposta.data.execucoes.some((item) => item.execucao_id === id)) {
        throw new Error('A execução selecionada não existe mais no servidor.')
      }
      atualRef.current = id
      setExecucaoId(id)
      await consultarStatus(id)
      if (solicitacao !== solicitacaoRef.current) return false
      localStorage.setItem(CHAVE_EXECUCAO, id)
      return true
    } catch (falha) {
      if (solicitacao !== solicitacaoRef.current) return false
      atualRef.current = ''
      setExecucaoId('')
      setStatusSemanal(null)
      localStorage.removeItem(CHAVE_EXECUCAO)
      setErro(obterMensagemErro(falha))
      return false
    } finally {
      if (solicitacao === solicitacaoRef.current) setCarregando(false)
    }
  }, [consultarStatus])

  const limparSelecao = useCallback(() => {
    ++solicitacaoRef.current
    atualRef.current = ''
    localStorage.removeItem(CHAVE_EXECUCAO)
    setExecucaoId('')
    setStatusSemanal(null)
    setErro('')
    setCarregando(false)
  }, [])

  const atualizarStatus = useCallback(async () => {
    const id = atualRef.current
    if (!id) return
    try {
      await consultarStatus(id)
      if (atualRef.current === id) setErro('')
    } catch (falha) {
      if (atualRef.current === id) setErro(obterMensagemErro(falha))
    }
  }, [consultarStatus])

  // Restauração inicial: as atualizações ocorrem nas respostas assíncronas.
  // O contador descarta respostas obsoletas (inclusive no StrictMode).
  useEffect(() => {
    if (!idInicial) return
    const referenciaSolicitacao = solicitacaoRef
    const solicitacao = ++referenciaSolicitacao.current

    void api.get<RespostaExecucoes>('/execucoes/').then(async ({ data }) => {
      if (solicitacao !== solicitacaoRef.current) return
      if (!data.execucoes.some((item) => item.execucao_id === idInicial)) {
        throw new Error('O histórico salvo não existe mais no servidor.')
      }
      const resposta = await api.get<StatusSemanal>(`/treinamentos-semanais/${idInicial}`)
      if (solicitacao !== solicitacaoRef.current) return
      atualRef.current = idInicial
      setExecucaoId(idInicial)
      setStatusSemanal(resposta.data)
      setErro('')
    }).catch((falha: unknown) => {
      if (solicitacao !== solicitacaoRef.current) return
      atualRef.current = ''
      setExecucaoId('')
      setStatusSemanal(null)
      localStorage.removeItem(CHAVE_EXECUCAO)
      setErro(obterMensagemErro(falha))
    }).finally(() => {
      if (solicitacao === solicitacaoRef.current) setCarregando(false)
    })

    return () => {
      ++referenciaSolicitacao.current
    }
  }, [idInicial])

  useEffect(() => {
    if (!execucaoId || statusSemanal?.status !== 'em_andamento') return
    const intervalo = window.setInterval(() => {
      void atualizarStatus()
    }, 3000)
    return () => window.clearInterval(intervalo)
  }, [execucaoId, statusSemanal?.status, atualizarStatus])

  return (
    <ExecucaoContext.Provider value={{
      execucaoId,
      statusSemanal,
      carregando,
      erro,
      selecionarExecucao,
      limparSelecao,
      atualizarStatus,
    }}>
      {children}
    </ExecucaoContext.Provider>
  )
}
