"""Atualiza o histórico de execuções com modelo semanal concluído.

O dataset original, o modelo e a avaliação externa nunca são alterados.
Atualizações aceitam somente lojas existentes e datas posteriores.
"""

from io import BytesIO
from pathlib import Path
import json
import os
import threading
import uuid

import pandas as pd

from dados.scripts.template import NOMES, validar_template
from backend.app.services.treinamento_service import pasta_da_execucao

TAMANHO_MAXIMO_ATUALIZACAO = 10 * 1024 * 1024
CONTROLE_ATUALIZACAO = threading.Lock()
OBRIGATORIAS = {"loja_id", "data", "vendas"}


def ler_historico(pasta: Path) -> pd.DataFrame:
    """Lê o último histórico confirmado ou o dataset original."""
    atualizado = pasta / "historico_atualizado.csv"
    caminho = atualizado if atualizado.is_file() else pasta / "dataset.csv"
    return pd.read_csv(
        caminho,
        dtype={"loja_id": "string", "data": "string"},
    )


def possui_modelo_concluido(pasta: Path) -> bool:
    """Somente o modelo AutoML semanal habilita atualização operacional."""
    pasta_modelo = pasta / "multihorizonte"
    requeridos = (
        pasta_modelo / "modelo_multihorizonte.joblib",
        pasta_modelo / "relatorio.json",
        pasta_modelo / "previsoes_teste.csv",
    )
    if not all(caminho.is_file() for caminho in requeridos):
        return False

    status = pasta / "status_multihorizonte.json"
    if not status.is_file():
        # Execuções antigas treinadas diretamente no terminal precisam
        # de migração/validação antes do uso operacional.
        return False

    try:
        dados = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        raise ValueError("Status semanal ilegível.") from erro
    return dados.get("status") == "concluido"


def atualizar_historico(
    execucao_id: str,
    conteudo: bytes,
    nome_arquivo: str,
) -> dict:
    """Adiciona novos registros sem substituir os anteriores nem retreinar."""
    pasta = pasta_da_execucao(execucao_id)

    if not possui_modelo_concluido(pasta):
        raise ValueError("Conclua o AutoML semanal antes da atualização.")
    if not nome_arquivo.lower().endswith(".csv"):
        raise ValueError("A atualização deve ser um CSV.")
    if not conteudo or len(conteudo) > TAMANHO_MAXIMO_ATUALIZACAO:
        raise ValueError("O CSV deve conter dados e ter até 10 MB.")

    try:
        novos = pd.read_csv(
            BytesIO(conteudo),
            dtype={"loja_id": "string", "data": "string"},
            encoding="utf-8-sig",
        )
    except (UnicodeError, ValueError, pd.errors.ParserError) as erro:
        raise ValueError("CSV de atualização inválido.") from erro

    if novos.empty:
        raise ValueError("O CSV de atualização está vazio.")
    faltantes = OBRIGATORIAS - set(novos.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    desconhecidas = set(novos.columns) - set(NOMES)
    if desconhecidas:
        raise ValueError(f"Colunas não reconhecidas: {sorted(desconhecidas)}")

    with CONTROLE_ATUALIZACAO:
        # Outra verificação após o bloqueio evita atualizar um estado mudado.
        if not possui_modelo_concluido(pasta):
            raise ValueError("O treinamento semanal não está concluído.")
        historico = ler_historico(pasta)
        colunas_historico = historico.columns.tolist()
        extras = set(novos.columns) - set(colunas_historico)
        if extras:
            raise ValueError(
                "Colunas inexistentes no histórico original: "
                f"{sorted(extras)}"
            )
        for coluna in colunas_historico:
            if coluna not in novos.columns:
                novos[coluna] = pd.NA
        novos = novos[colunas_historico].copy()

        verificacao = validar_template(novos, min_linhas=1, min_dias=1)
        if not verificacao.valido:
            raise ValueError("Atualização inválida: " + "; ".join(verificacao.erros))

        lojas = set(historico["loja_id"].astype(str))
        desconhecidas = set(novos["loja_id"].astype(str)) - lojas
        if desconhecidas:
            raise ValueError(f"Lojas desconhecidas: {sorted(desconhecidas)}")

        antigas_datas = pd.to_datetime(
            historico["data"], format="%Y-%m-%d", errors="raise"
        )
        novas_datas = pd.to_datetime(
            novos["data"], format="%Y-%m-%d", errors="raise"
        )
        ultimas = historico.assign(_data=antigas_datas).groupby("loja_id")["_data"].max()
        for loja, grupo in novos.assign(_data=novas_datas).groupby("loja_id"):
            if (grupo["_data"] <= ultimas.loc[loja]).any():
                raise ValueError(
                    f"A loja {loja} possui datas anteriores ou iguais "
                    f"a {ultimas.loc[loja].date()}."
                )

        combinado = pd.concat([historico, novos], ignore_index=True)
        combinado = combinado.sort_values(["loja_id", "data"]).reset_index(drop=True)
        validacao = validar_template(combinado, min_linhas=90, min_dias=90)
        if not validacao.valido:
            raise ValueError("Histórico resultante inválido: " + "; ".join(validacao.erros))

        destino = pasta / "historico_atualizado.csv"
        temporario = pasta / f".historico_atualizado.{uuid.uuid4().hex}.tmp"
        try:
            combinado.to_csv(temporario, index=False)
            os.replace(temporario, destino)
        finally:
            temporario.unlink(missing_ok=True)

        return {
            "execucao_id": pasta.name,
            "status": "historico_atualizado",
            "registros_adicionados": int(len(novos)),
            "registros_totais": int(len(combinado)),
            "quantidade_lojas": int(combinado["loja_id"].nunique()),
            "ultimas_datas_por_loja": combinado.groupby("loja_id")["data"].max().to_dict(),
            "modelo_retreinado": False,
            "avaliacao_externa_preservada": True,
        }
