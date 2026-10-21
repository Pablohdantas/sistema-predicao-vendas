
import { useState } from 'react'
import {
  ChevronDown,
  ChevronUp,
  Download,
  FileSpreadsheet,
} from 'lucide-react'

import { baixarTemplateCSV } from '../utils/baixarTemplate'

const colunas = [
  ['loja_id', 'Obrigatória', 'Número inteiro positivo que identifica a loja.'],
  ['data', 'Obrigatória', 'Data da venda no formato AAAA-MM-DD.'],
  ['vendas', 'Obrigatória', 'Valor ou quantidade de vendas, maior ou igual a zero.'],
  ['promocao', 'Opcional', '1 se houve promoção; 0 caso contrário.'],
  ['promocao_continua', 'Opcional', '1 se havia promoção contínua ativa; 0 caso contrário.'],
  ['mes_renovacao_promocao', 'Opcional', '1 se a promoção contínua é renovada naquele mês.'],
  ['feriado', 'Opcional', '1 se o dia era feriado oficial; 0 caso contrário.'],
  ['feriado_escolar', 'Opcional', '1 se era feriado escolar; 0 caso contrário.'],
  ['distancia_concorrente', 'Opcional', 'Distância, em metros, ao concorrente mais próximo.'],
  ['tipo_loja', 'Opcional', 'Categoria que identifica o formato da loja.'],
  ['sortimento', 'Opcional', 'Categoria que identifica o nível de sortimento.'],
] as const

export default function GuiaTemplate() {
  const [mostrarColunas, setMostrarColunas] = useState(false)

  return (
    <div className="guia-template">
      <div className="guia-template-intro">
        <FileSpreadsheet size={19} />
        <div>
          <strong>Como preparar seu CSV?</strong>
          <p>
            Escolha o modelo básico ou o completo.
            Os dois são aceitos pela ferramenta.
          </p>
        </div>
      </div>

      <div className="guia-template-acoes">
        <button
          type="button"
          className="secondary-button"
          onClick={() => baixarTemplateCSV('basico')}
        >
          <Download size={17} />
          Modelo básico
        </button>

        <button
          type="button"
          className="secondary-button"
          onClick={() => baixarTemplateCSV('completo')}
        >
          <Download size={17} />
          Modelo completo
        </button>
      </div>

      <p className="template-informacao">
        O modelo básico utiliza apenas loja, data e vendas.
        O completo inclui oito colunas opcionais.
        Os arquivos baixados contêm somente exemplos:
        substitua os registros por pelo menos 90 dias
        distintos de histórico.
      </p>

      <button
        type="button"
        className="guia-template-expandir"
        onClick={() => setMostrarColunas(!mostrarColunas)}
        aria-expanded={mostrarColunas}
      >
        Entender todas as colunas
        {mostrarColunas
          ? <ChevronUp size={17} />
          : <ChevronDown size={17} />}
      </button>

      {mostrarColunas && (
        <div className="guia-template-tabela-wrapper">
          <table className="guia-template-tabela">
            <thead>
              <tr>
                <th>Coluna</th>
                <th>Tipo</th>
                <th>O que significa?</th>
              </tr>
            </thead>

            <tbody>
              {colunas.map(([nome, tipo, descricao]) => (
                <tr key={nome}>
                  <td><code>{nome}</code></td>
                  <td>{tipo}</td>
                  <td>{descricao}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
