import { createContext } from 'react'

export const CHAVE_EXECUCAO = 'automl_execucao_atual'

type Metricas = {
  rmse: number
  mae: number
  quantidade: number
}

export type StatusSemanal = {
  execucao_id: string
  status: 'nao_iniciado' | 'em_andamento' | 'concluido' | 'falhou' | 'interrompido'
  algoritmo_vencedor?: string
  avaliacao_externa?: {
    agregado: Metricas
    por_horizonte: Record<string, Metricas>
  }
  erro?: string | null
}

export type Contexto = {
  execucaoId: string
  statusSemanal: StatusSemanal | null
  carregando: boolean
  erro: string
  selecionarExecucao: (id: string) => Promise<boolean>
  limparSelecao: () => void
  atualizarStatus: () => Promise<void>
}

export const ExecucaoContext = createContext<Contexto | null>(null)
