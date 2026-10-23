import { useContext } from 'react'
import { ExecucaoContext, type Contexto } from './execucaoContext'

export function useExecucao(): Contexto {
  const contexto = useContext(ExecucaoContext)
  if (!contexto) throw new Error('useExecucao requer ExecucaoProvider.')
  return contexto
}
