import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react'
import type { ReactNode } from 'react'
import { api, obterMensagemErro } from '../services/api'

const CHAVE_EXECUCAO = 'automl_execucao_atual'

type Metricas = {
  rmse: number
  mae: number
  quantidade: number
}

type StatusSemanal = {
  execucao_id: string
  status: 'nao_iniciado' | 'em_andamento' | 'concluido' | 'falhou' | 'interrompido'
  algoritmo_vencedor?: string
  avaliacao_externa?: {
    agregado: Metricas
    por_horizonte: Record<string, Metricas>
  }
  erro?: string | null
}

type RespostaExecucoes = {
  execucoes: { execucao_id: string }[]
}

type Contexto = {
  execucaoId: string
  statusSemanal: StatusSemanal | null
  carregando: boolean
  erro: string
  selecionarExecucao: (id: string) => Promise<boolean>
  limparSelecao: () => void
  atualizarStatus: () => Promise<void>
}

const ExecucaoContext = createContext<Contexto | null>(null)

export function ExecucaoProvider({ children }: { children: ReactNode }) {
  const [execucaoId, setExecucaoId] = useState('')
  const [statusSemanal, setStatusSemanal] = useState<StatusSemanal | null>(null)
  const [carregando, setCarregando] = useState(true)
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

  useEffect(() => {
    const idSalvo = localStorage.getItem(CHAVE_EXECUCAO)
    if (idSalvo) {
      void selecionarExecucao(idSalvo)
    } else {
      setCarregando(false)
    }
  }, [selecionarExecucao])

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

export function useExecucao(): Contexto {
  const contexto = useContext(ExecucaoContext)
  if (!contexto) throw new Error('useExecucao requer ExecucaoProvider.')
  return contexto
}
