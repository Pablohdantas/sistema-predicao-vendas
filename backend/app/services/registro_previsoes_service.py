"""Registro imutável de previsões operacionais semanais por loja e origem.

Os arquivos são criados exclusivamente (sem sobrescrever resultados anteriores).
Não armazena modelos ou dados originais; usa apenas a pasta validada da execução.
"""

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from backend.app.services.treinamento_service import pasta_da_execucao


PASTA_REGISTROS = "previsoes_registradas"


def _diretorio(execucao_id: str) -> Path:
    return pasta_da_execucao(execucao_id) / PASTA_REGISTROS


def _caminho(execucao_id: str, loja_id: int, data_origem: str) -> Path:
    if not isinstance(loja_id, int) or loja_id <= 0:
        raise ValueError("Identificador da loja inválido.")
    try:
        origem = datetime.strptime(data_origem, "%Y-%m-%d").date()
    except (ValueError, TypeError) as erro:
        raise ValueError("Data de origem inválida.") from erro
    if origem.isoformat() != data_origem:
        raise ValueError("Data de origem inválida.")
    return _diretorio(execucao_id) / f"loja_{loja_id}_{data_origem}.json"


def consultar_previsao_registrada(execucao_id: str, loja_id: int, data_origem: str) -> dict | None:
    caminho = _caminho(execucao_id, loja_id, data_origem)
    if not caminho.is_file():
        return None
    with caminho.open("r", encoding="utf-8") as arquivo:
        registro = json.load(arquivo)
    if not isinstance(registro, dict) or registro.get("data_origem") != data_origem or registro.get("loja_id") != loja_id:
        raise ValueError("Registro de previsão inválido.")
    return registro


def registrar_previsao(execucao_id: str, previsao: dict) -> dict:
    """Grava somente se ainda não existir; requisições repetidas recebem o original."""
    loja_id = previsao.get("loja_id")
    origem = previsao.get("data_origem")
    destino = _caminho(execucao_id, loja_id, origem)
    valores = previsao.get("previsoes")
    if not isinstance(valores, list) or len(valores) != 7:
        raise ValueError("Registro deve conter sete previsões.")
    # Normaliza o formato interno (data) para o contrato público
    # (data_prevista) usado pela rota de previsão e pelo frontend.
    normalizados = []
    for horizonte, item in enumerate(valores, start=1):
        if not isinstance(item, dict) or item.get("horizonte_dias") != horizonte:
            raise ValueError("Horizontes de previsão inválidos.")
        data = item.get("data_prevista", item.get("data"))
        try:
            valor = float(item["vendas_previstas"])
            data_validada = datetime.strptime(data, "%Y-%m-%d").date()
        except (KeyError, TypeError, ValueError) as erro:
            raise ValueError("Data ou valor previsto inválido.") from erro
        if data_validada.isoformat() != data:
            raise ValueError("Data prevista inválida.")
        origem_data = datetime.strptime(origem, "%Y-%m-%d").date()
        if (data_validada - origem_data).days != horizonte:
            raise ValueError("Data prevista incompatível com o horizonte.")
        if not math.isfinite(valor) or valor < 0:
            raise ValueError("Previsões não finitas ou negativas.")
        normalizados.append({
            "horizonte_dias": horizonte,
            "data_prevista": data,
            "vendas_previstas": valor,
        })

    existente = consultar_previsao_registrada(execucao_id, loja_id, origem)
    if existente is not None:
        return existente

    destino.parent.mkdir(parents=True, exist_ok=True)
    registro = {
        **previsao,
        "previsoes": normalizados,
        "registrado_em": datetime.now(timezone.utc).isoformat(),
    }
    temporario = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=destino.parent, prefix=".previsao_", suffix=".tmp", encoding="utf-8", delete=False) as arquivo:
            temporario = Path(arquivo.name)
            json.dump(registro, arquivo, ensure_ascii=False, indent=2, allow_nan=False)
            arquivo.flush()
            os.fsync(arquivo.fileno())
        # hard link é atômico e falha se o destino já existir: não substitui previsão.
        try:
            os.link(temporario, destino)
        except FileExistsError:
            pass
    finally:
        if temporario is not None:
            temporario.unlink(missing_ok=True)

    salvo = consultar_previsao_registrada(execucao_id, loja_id, origem)
    if salvo is None:
        raise RuntimeError("Falha ao registrar previsão.")
    return salvo
