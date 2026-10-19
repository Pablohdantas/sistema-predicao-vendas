# Dicionário de dados do template

Uma linha representa **uma loja em um dia em que ela operou**. Dias de loja fechada não devem ser enviados.

| Coluna | Tipo | Obrigatória | Descrição |
|---|---|---|---|
| `loja_id` | identificador | sim | Identificador da loja/unidade |
| `data` | data (AAAA-MM-DD) | sim | Data no formato AAAA-MM-DD |
| `vendas` | número | sim | VARIÁVEL-ALVO: vendas do dia (>= 0) |
| `promocao` | 0 ou 1 | não | 1 se havia promoção no dia |
| `promocao_continua` | 0 ou 1 | não | 1 se a loja já aderiu à promoção contínua (tipo Promo2) na data |
| `mes_renovacao_promocao` | 0 ou 1 | não | 1 se a loja participa da promoção contínua e o mês é de início de nova rodada |
| `feriado` | 0 ou 1 | não | 1 se o dia era feriado oficial |
| `feriado_escolar` | 0 ou 1 | não | 1 se o dia era feriado escolar |
| `distancia_concorrente` | número | não | Distância ao concorrente mais próximo (m); vazio se desconhecida |
| `tipo_loja` | texto (categoria) | não | Tipo/formato da loja |
| `sortimento` | texto (categoria) | não | Nível de sortimento de produtos |

## Regras de aceitação

- Frequência **diária** (mediana do intervalo entre registros de uma loja de até 3 dias).
- Mínimo de **1000 linhas** no total e de **180 dias distintos** de histórico.
- Lojas com menos de 56 dias de histórico geram aviso (podem ser desconsideradas na modelagem).
- Sem linhas duplicadas para o par (`loja_id`, `data`).
- `vendas` não pode ser negativa; colunas binárias aceitam apenas 0 ou 1.
- Colunas opcionais podem ser omitidas; `distancia_concorrente` pode ter células vazias.
- Colunas fora do template são ignoradas.
