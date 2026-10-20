
"""
Armazenamento local de datasets aprovados.

Cada upload válido cria uma execução independente.
Não inicia treinamento AutoML.
"""

import json
import shutil
import uuid

from pathlib import Path


RAIZ = Path(__file__).resolve().parents[3]
PASTA_EXECUCOES = RAIZ / "backend" / "execucoes"


def salvar_execucao(
    conteudo: bytes,
    relatorio: dict
) -> dict:
    if not relatorio.get("valido"):
        raise ValueError(
            "Somente datasets válidos podem ser armazenados."
        )

    identificador = uuid.uuid4().hex
    pasta = PASTA_EXECUCOES / identificador

    pasta.mkdir(
        parents=True,
        exist_ok=False
    )

    try:
        # Não utilizamos diretamente o nome enviado
        # pelo usuário para construir caminhos.
        arquivo_csv = pasta / "dataset.csv"
        arquivo_relatorio = pasta / "validacao.json"

        arquivo_csv.write_bytes(conteudo)

        dados_relatorio = {
            **relatorio,
            "execucao_id": identificador,
            "status": "aguardando_treinamento"
        }

        arquivo_relatorio.write_text(
            json.dumps(
                dados_relatorio,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        return dados_relatorio

    except Exception:
        # Não deixar execuções parcialmente gravadas.
        shutil.rmtree(pasta)
        raise
