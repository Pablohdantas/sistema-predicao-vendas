
export type ParametrosTreinamento = {
  modo_teste: false
  tentativas: number
  tempo_por_algoritmo: number
}

export type PerfilTreinamento =
  | 'rapido'
  | 'ampliado'
  | 'personalizado'

export const PERFIS_TREINAMENTO: Record<
  Exclude<PerfilTreinamento, 'personalizado'>,
  ParametrosTreinamento
> = {
  rapido: {
    modo_teste: false,
    tentativas: 2,
    tempo_por_algoritmo: 60,
  },
  ampliado: {
    modo_teste: false,
    tentativas: 15,
    tempo_por_algoritmo: 360,
  },
}

export function parametrosValidos(
  valor: ParametrosTreinamento,
): boolean {
  return (
    Number.isInteger(valor.tentativas) &&
    valor.tentativas >= 1 &&
    valor.tentativas <= 30 &&
    Number.isInteger(valor.tempo_por_algoritmo) &&
    valor.tempo_por_algoritmo >= 1 &&
    valor.tempo_por_algoritmo <= 1800
  )
}
