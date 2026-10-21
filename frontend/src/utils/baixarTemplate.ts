
type TipoTemplate = 'basico' | 'completo'

const CABECALHO_BASICO = [
  'loja_id',
  'data',
  'vendas',
]

const CABECALHO_COMPLETO = [
  ...CABECALHO_BASICO,
  'promocao',
  'promocao_continua',
  'mes_renovacao_promocao',
  'feriado',
  'feriado_escolar',
  'distancia_concorrente',
  'tipo_loja',
  'sortimento',
]

const EXEMPLO_BASICO = [
  '1,2025-01-01,1200.50',
  '1,2025-01-02,1350.00',
  '1,2025-01-03,1280.75',
]

const EXEMPLO_COMPLETO = [
  '1,2025-01-01,1200.50,0,0,0,1,0,350,a,a',
  '1,2025-01-02,1350.00,1,0,0,0,0,350,a,a',
  '1,2025-01-03,1280.75,0,0,0,0,0,350,a,a',
]

export function baixarTemplateCSV(
  tipo: TipoTemplate = 'basico',
): void {
  const completo = tipo === 'completo'

  const linhas = completo
    ? [CABECALHO_COMPLETO.join(','), ...EXEMPLO_COMPLETO]
    : [CABECALHO_BASICO.join(','), ...EXEMPLO_BASICO]

  const conteudo = '\uFEFF' + linhas.join('\n') + '\n'

  const arquivo = new Blob([conteudo], {
    type: 'text/csv;charset=utf-8',
  })

  const url = URL.createObjectURL(arquivo)
  const link = document.createElement('a')

  link.href = url
  link.download = completo
    ? 'template_vendas_completo.csv'
    : 'template_vendas_basico.csv'

  document.body.appendChild(link)
  link.click()
  link.remove()

  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
