"""
Gera a documentação do template a partir do próprio esquema (fonte única da verdade):
  - dados/documentacao/template_modelo.csv   (arquivo de exemplo para o usuário)
  - dados/documentacao/dicionario_template.md (dicionário de dados + regras de aceitação)

Uso (na raiz do projeto, com a .venv ativada):
    python dados/scripts/gerar_documentacao.py
"""
from pathlib import Path

import pandas as pd

from template import (COLUNAS, MAX_MEDIANA_INTERVALO_DIAS, MIN_DIAS_DISTINTOS,
                      MIN_DIAS_HISTORICO, MIN_LINHAS_TOTAL, NOMES, validar_template)

DOC = Path(__file__).resolve().parents[1] / "documentacao"
DOC.mkdir(exist_ok=True)

# ---------------------------------------------------------------- template modelo
datas = pd.date_range("2025-03-03", periods=6, freq="D")
linhas = []
for loja, base, tipo in [(1, 5200.0, "a"), (2, 3100.0, "b")]:
    for i, d in enumerate(datas):
        linhas.append({
            "loja_id": loja, "data": d.strftime("%Y-%m-%d"), "vendas": base + 150 * i,
            "promocao": int(i % 3 == 0), "promocao_continua": 0, "mes_renovacao_promocao": 0,
            "feriado": 0, "feriado_escolar": 0,
            "distancia_concorrente": 1200 if loja == 1 else "", "tipo_loja": tipo, "sortimento": "a",
        })
modelo = pd.DataFrame(linhas)[NOMES]
modelo.to_csv(DOC / "template_modelo.csv", index=False)

# ---------------------------------------------------------------- dicionário
ROTULO = {"id": "identificador", "data": "data (AAAA-MM-DD)", "numero": "número",
          "binario": "0 ou 1", "categoria": "texto (categoria)"}
md = ["# Dicionário de dados do template", "",
      "Uma linha representa **uma loja em um dia em que ela operou**. Dias de loja fechada não devem ser enviados.", "",
      "| Coluna | Tipo | Obrigatória | Descrição |", "|---|---|---|---|"]
for c in COLUNAS:
    md.append(f"| `{c.nome}` | {ROTULO[c.tipo]} | {'sim' if c.obrigatoria else 'não'} | {c.descricao} |")
md += ["", "## Regras de aceitação", "",
       f"- Frequência **diária** (mediana do intervalo entre registros de uma loja de até {MAX_MEDIANA_INTERVALO_DIAS} dias).",
       f"- Mínimo de **{MIN_LINHAS_TOTAL} linhas** no total e de **{MIN_DIAS_DISTINTOS} dias distintos** de histórico.",
       f"- Lojas com menos de {MIN_DIAS_HISTORICO} dias de histórico geram aviso (podem ser desconsideradas na modelagem).",
       "- Sem linhas duplicadas para o par (`loja_id`, `data`).",
       "- `vendas` não pode ser negativa; colunas binárias aceitam apenas 0 ou 1.",
       "- Colunas opcionais podem ser omitidas; `distancia_concorrente` pode ter células vazias.",
       "- Colunas fora do template são ignoradas.", ""]
(DOC / "dicionario_template.md").write_text("\n".join(md), encoding="utf-8")

print("Gerado em:", DOC)
print("O template modelo é válido?", validar_template(modelo).valido,
      "(esperado: False, pois é só um exemplo curto e viola os mínimos de tamanho)")