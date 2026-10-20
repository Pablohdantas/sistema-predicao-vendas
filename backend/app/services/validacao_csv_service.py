
"""
Serviço de validação de CSV.

Reutiliza o contrato e o validador do template
já implementados em dados/scripts/template.py.

Esta etapa apenas valida o conteúdo recebido.
Não executa AutoML nem altera os dados originais.
"""

from io import BytesIO

import pandas as pd

from dados.scripts.template import (
    COLUNAS,
    validar_template
)


TAMANHO_MAXIMO_BYTES = 100 * 1024 * 1024


def validar_csv(conteudo: bytes, nome_arquivo: str):
    """
    Recebe os bytes de um arquivo CSV e retorna
    um relatório estruturado de validação.
    """

    if not nome_arquivo.lower().endswith(".csv"):
        raise ValueError(
            "O arquivo precisa ter extensão .csv."
        )

    if not conteudo:
        raise ValueError(
            "O arquivo enviado está vazio."
        )

    if len(conteudo) > TAMANHO_MAXIMO_BYTES:
        raise ValueError(
            "Arquivo maior que o limite de 100 MB."
        )

    try:
        df = pd.read_csv(
            BytesIO(conteudo),
            dtype={
                "loja_id": "string",
                "data": "string"
            },
            encoding="utf-8-sig"
        )
    except (UnicodeError, pd.errors.ParserError, ValueError) as erro:
        raise ValueError(
            f"Não foi possível ler o CSV: {erro}"
        ) from erro

    resultado = validar_template(df)

    relatorio = {
        "nome_arquivo": nome_arquivo,
        "valido": resultado.valido,
        "quantidade_linhas": int(len(df)),
        "quantidade_colunas": int(len(df.columns)),
        "colunas_recebidas": df.columns.tolist(),
        "colunas_obrigatorias": [
            coluna.nome
            for coluna in COLUNAS
            if coluna.obrigatoria
        ],
        "colunas_opcionais": [
            coluna.nome
            for coluna in COLUNAS
            if not coluna.obrigatoria
        ],
        "erros": resultado.erros,
        "avisos": resultado.avisos
    }

    if resultado.valido:
        relatorio["quantidade_lojas"] = int(
            df["loja_id"].nunique()
        )

        relatorio["quantidade_dias"] = int(
            df["data"].nunique()
        )

        relatorio["periodo"] = {
            "inicio": str(df["data"].min()),
            "fim": str(df["data"].max())
        }

    return relatorio
