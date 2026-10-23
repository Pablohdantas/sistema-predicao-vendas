"""Lista históricos e apresenta o estado do AutoML multihorizonte.

Não carrega modelos serializados e não modifica as execuções.
Compatível com o contrato do seletor do frontend.
"""

import json
import re
from pathlib import Path

from fastapi import APIRouter

from backend.app.services.execucoes_service import PASTA_EXECUCOES

router = APIRouter(prefix="/execucoes", tags=["Execuções"])
ID_VALIDO = re.compile(r"^[a-f0-9]{32}$")
ARTEFATOS_OBRIGATORIOS = (
    "modelo_multihorizonte.joblib",
    "relatorio.json",
    "previsoes_teste.csv",
    "previsoes_futuras.json",
)


def ler_json(caminho: Path) -> dict:
    if not caminho.is_file():
        return {}
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            valor = json.load(arquivo)
        return valor if isinstance(valor, dict) else {}
    except (OSError, ValueError):
        return {}


def modelo_semanal_disponivel(pasta: Path) -> bool:
    destino = pasta / "multihorizonte"
    if not all((destino / nome).is_file() for nome in ARTEFATOS_OBRIGATORIOS):
        return False
    relatorio = ler_json(destino / "relatorio.json")
    return (
        relatorio.get("tipo") == "automl_multihorizonte"
        and relatorio.get("algoritmo_vencedor") in ("lgbm", "xgboost")
        and isinstance(relatorio.get("avaliacao_externa"), dict)
    )


def estado_semanal(pasta: Path) -> tuple[str, str | None]:
    status = ler_json(pasta / "status_multihorizonte.json")
    valor = status.get("status")
    pronto = modelo_semanal_disponivel(pasta)

    if valor == "concluido":
        if pronto:
            return "concluido", status.get("algoritmo_vencedor")
        return "interrompido", None

    if valor == "em_andamento":
        # A rota de treinamento é responsável por detectar interrupções.
        return "em_andamento", None

    if valor in ("falhou", "interrompido"):
        return valor, None

    # Compatibilidade com execuções históricas sem status, mas completas.
    if pronto:
        relatorio = ler_json(pasta / "multihorizonte" / "relatorio.json")
        return "concluido", relatorio.get("algoritmo_vencedor")

    return "aguardando_treinamento", None


@router.get("/")
def listar_execucoes():
    if not PASTA_EXECUCOES.is_dir():
        return {"quantidade": 0, "execucoes": []}

    execucoes = []
    for pasta in PASTA_EXECUCOES.iterdir():
        if not pasta.is_dir() or not ID_VALIDO.fullmatch(pasta.name):
            continue
        if not (pasta / "dataset.csv").is_file():
            continue

        validacao = ler_json(pasta / "validacao.json")
        status, vencedor = estado_semanal(pasta)
        try:
            criado_em = pasta.stat().st_ctime
        except OSError:
            continue

        execucoes.append({
            "execucao_id": pasta.name,
            "nome_arquivo": validacao.get("nome_arquivo") or "Dataset sem nome",
            "quantidade_linhas": validacao.get("quantidade_linhas"),
            "quantidade_lojas": validacao.get("quantidade_lojas"),
            "quantidade_dias": validacao.get("quantidade_dias"),
            "periodo": validacao.get("periodo"),
            "status": status,
            "modo": "multihorizonte",
            "etapa": status,
            "algoritmo_vencedor": vencedor,
            "criado_em": criado_em,
            "iniciado_em": None,
            "finalizado_em": None,
        })

    execucoes.sort(key=lambda item: item["criado_em"], reverse=True)
    return {"quantidade": len(execucoes), "execucoes": execucoes}
