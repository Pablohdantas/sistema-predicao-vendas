
import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export type DatasetResponse = {
  valido: boolean
  execucao_id: string | null
  status: string
  nome_arquivo: string
  quantidade_linhas: number
  quantidade_lojas?: number
  quantidade_dias?: number
  erros: string[]
  avisos: string[]
}

export type AvaliacaoExterna = {
  algoritmo: string
  rmse: number
  mae: number
  quantidade_previsoes: number
  quantidade_lojas: number
  periodo_teste: {
    inicio: string
    fim: string
  }
}

export type ResultadoTreinamento = {
  melhor_algoritmo_na_validacao: string
  algoritmo_vencedor: string | null
  rmse_selecao: number
  mae_selecao: number
  modelo_salvo: string | null
  quantidade_dobras: number
  tempo_total_segundos: number
  avaliacao_externa?: AvaliacaoExterna | null
}

export type StatusTreinamento = {
  execucao_id: string
  status: string
  etapa: string | null
  modo: string | null
  erro?: string | null
  resultado: ResultadoTreinamento | null
}

export type PrevisaoExterna = {
  loja_id: number
  data_origem: string
  data_prevista: string
  vendas_reais: number
  vendas_previstas: number
  erro: number
  erro_absoluto: number
}

export type ListaPrevisoes = {
  execucao_id: string
  quantidade_retornada: number
  previsoes: PrevisaoExterna[]
}

export function obterMensagemErro(erro: unknown): string {
  if (axios.isAxiosError(erro)) {
    if (!erro.response) {
      return (
        'Não foi possível conectar ao backend. ' +
        'Verifique se o FastAPI está em execução na porta 8000.'
      )
    }

    const detalhe: unknown = erro.response.data?.detail

    if (typeof detalhe === 'string') {
      return detalhe
    }

    if (erro.response.status === 409) {
      return 'Já existe um treinamento em andamento ou concluído.'
    }

    return `Erro HTTP ${erro.response.status}.`
  }

  if (erro instanceof Error) {
    return erro.message
  }

  return 'Ocorreu um erro inesperado.'
}
