
"""
Listagem das execuções AutoML armazenadas localmente.

Não carrega datasets completos nem modelos em memória.
Consulta apenas os relatórios JSON de cada execução.
"""

import json
from pathlib import Path

from fastapi import APIRouter

from backend.app.services.execucoes_service import PASTA_EXECUCOES


router = APIRouter(
    prefix="/execucoes",
    tags=["Execuções"]
)


def ler_json(caminho: Path) -> dict:
    if not caminho.is_file():
        return {}

    try:
        with caminho.open(encoding="utf-8") as arquivo:
            dados = json.load(arquivo)

        return dados if isinstance(dados, dict) else {}

    except (OSError, ValueError):
        return {}


@router.get("/")
def listar_execucoes():
    """
    Lista execuções com dataset salvo.

    Não inicia treinamentos nem modifica os arquivos.
    """

    if not PASTA_EXECUCOES.is_dir():
        return {"quantidade": 0, "execucoes": []}

    execucoes = []

    for pasta in PASTA_EXECUCOES.iterdir():
        if not pasta.is_dir():
            continue

        # Apenas diretórios com identificadores válidos.
        if (
            len(pasta.name) != 32
            or any(
                caractere not in "0123456789abcdef"
                for caractere in pasta.name
            )
        ):
            continue

        if not (pasta / "dataset.csv").is_file():
            continue

        validacao = ler_json(pasta / "validacao.json")
        treinamento = ler_json(
            pasta / "status_treinamento.json"
        )

        status = treinamento.get(
            "status",
            validacao.get("status", "aguardando_treinamento")
        )

        resultado = treinamento.get("resultado") or {}

        execucoes.append({
            "execucao_id": pasta.name,
            "nome_arquivo": validacao.get(
                "nome_arquivo",
                "Dataset sem nome"
            ),
            "quantidade_linhas": validacao.get(
                "quantidade_linhas"
            ),
            "quantidade_lojas": validacao.get(
                "quantidade_lojas"
            ),
            "quantidade_dias": validacao.get(
                "quantidade_dias"
            ),
            "periodo": validacao.get("periodo"),
            "status": status,
            "modo": treinamento.get("modo"),
            "etapa": treinamento.get("etapa"),
            "algoritmo_vencedor": resultado.get(
                "algoritmo_vencedor"
            ),
            "criado_em": pasta.stat().st_ctime,
            "iniciado_em": treinamento.get("iniciado_em"),
            "finalizado_em": treinamento.get("finalizado_em")
        })

    execucoes.sort(
        key=lambda item: item["criado_em"],
        reverse=True
    )

    return {
        "quantidade": len(execucoes),
        "execucoes": execucoes
    }
