
"""
Atualização do histórico de vendas de uma execução.

Preserva dataset.csv, o modelo treinado e os resultados
do teste externo.

As novas vendas são acrescentadas ao arquivo
historico_atualizado.csv, sem retreinamento.
"""

from io import BytesIO
from pathlib import Path
import os
import threading

import pandas as pd

from dados.scripts.template import (
    NOMES,
    validar_template
)
from backend.app.services.treinamento_service import (
    consultar_treinamento,
    pasta_da_execucao
)


TAMANHO_MAXIMO_ATUALIZACAO = 10 * 1024 * 1024
CONTROLE_ATUALIZACAO = threading.Lock()

OBRIGATORIAS = {"loja_id", "data", "vendas"}


def ler_historico(pasta: Path) -> pd.DataFrame:
    """
    Carrega o histórico atualizado, caso exista,
    ou o dataset original da execução.
    """

    atualizado = pasta / "historico_atualizado.csv"
    caminho = (
        atualizado
        if atualizado.is_file()
        else pasta / "dataset.csv"
    )

    return pd.read_csv(
        caminho,
        dtype={
            "loja_id": "string",
            "data": "string"
        }
    )


def atualizar_historico(
    execucao_id: str,
    conteudo: bytes,
    nome_arquivo: str
) -> dict:
    """
    Acrescenta novas vendas a lojas existentes.

    Só aceita datas posteriores à última observação
    de cada loja, para evitar alterações retroativas.
    """

    pasta = pasta_da_execucao(execucao_id)

    status = consultar_treinamento(pasta.name)

    if (
        status.get("status") != "concluido"
        or status.get("modo") != "treinamento_definitivo"
    ):
        raise ValueError(
            "O histórico só pode ser atualizado após "
            "um treinamento definitivo concluído."
        )

    if not nome_arquivo.lower().endswith(".csv"):
        raise ValueError(
            "A atualização deve ser um arquivo CSV."
        )

    if not conteudo:
        raise ValueError(
            "O arquivo de atualização está vazio."
        )

    if len(conteudo) > TAMANHO_MAXIMO_ATUALIZACAO:
        raise ValueError(
            "A atualização não pode ultrapassar 10 MB."
        )

    try:
        novos = pd.read_csv(
            BytesIO(conteudo),
            dtype={
                "loja_id": "string",
                "data": "string"
            },
            encoding="utf-8-sig"
        )
    except (
        UnicodeError,
        ValueError,
        pd.errors.ParserError
    ) as erro:
        raise ValueError(
            f"Não foi possível ler a atualização: {erro}"
        ) from erro

    if novos.empty:
        raise ValueError(
            "A atualização não contém registros."
        )

    faltantes = OBRIGATORIAS - set(novos.columns)

    if faltantes:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(faltantes)}"
        )

    desconhecidas = set(novos.columns) - set(NOMES)

    if desconhecidas:
        raise ValueError(
            "Colunas não reconhecidas: "
            f"{sorted(desconhecidas)}"
        )

    with CONTROLE_ATUALIZACAO:
        historico = ler_historico(pasta)

        # Mantém exatamente o esquema original.
        colunas_historico = historico.columns.tolist()

        extras = set(novos.columns) - set(colunas_historico)

        if extras:
            raise ValueError(
                "A atualização contém colunas que não "
                "existiam no dataset original: "
                f"{sorted(extras)}"
            )

        for coluna in colunas_historico:
            if coluna not in novos.columns:
                novos[coluna] = pd.NA

        novos = novos[colunas_historico].copy()

        # Antes de acrescentar, validar os novos
        # registros de forma independente.
        verificacao_novos = validar_template(
            novos,
            min_linhas=1,
            min_dias=1
        )

        if not verificacao_novos.valido:
            raise ValueError(
                "Atualização inválida: "
                + "; ".join(verificacao_novos.erros)
            )

        ids_existentes = set(
            historico["loja_id"].astype(str)
        )

        ids_novos = set(
            novos["loja_id"].astype(str)
        )

        lojas_desconhecidas = (
            ids_novos - ids_existentes
        )

        if lojas_desconhecidas:
            raise ValueError(
                "A atualização contém lojas desconhecidas: "
                f"{sorted(lojas_desconhecidas)}"
            )

        datas_historico = pd.to_datetime(
            historico["data"],
            format="%Y-%m-%d",
            errors="raise"
        )

        datas_novas = pd.to_datetime(
            novos["data"],
            format="%Y-%m-%d",
            errors="raise"
        )

        ultimas_datas = (
            historico.assign(_data=datas_historico)
            .groupby("loja_id")["_data"]
            .max()
        )

        for loja_id, grupo in novos.assign(
            _data=datas_novas
        ).groupby("loja_id"):
            ultima_data = ultimas_datas.loc[loja_id]

            if (grupo["_data"] <= ultima_data).any():
                raise ValueError(
                    f"A loja {loja_id} possui registros "
                    "com datas anteriores ou iguais "
                    "à última venda armazenada "
                    f"({ultima_data.date()})."
                )

        combinado = pd.concat(
            [historico, novos],
            ignore_index=True
        )

        combinado = combinado.sort_values(
            ["loja_id", "data"]
        ).reset_index(drop=True)

        # Revalida o histórico completo.
        validacao_final = validar_template(
            combinado,
            min_linhas=90,
            min_dias=90
        )

        if not validacao_final.valido:
            raise ValueError(
                "Histórico resultante inválido: "
                + "; ".join(validacao_final.erros)
            )

        destino = pasta / "historico_atualizado.csv"
        temporario = pasta / ".historico_atualizado.tmp"

        try:
            combinado.to_csv(
                temporario,
                index=False
            )

            os.replace(temporario, destino)

        finally:
            temporario.unlink(missing_ok=True)

        ultimas = (
            combinado.groupby("loja_id")["data"]
            .max()
            .to_dict()
        )

        return {
            "execucao_id": pasta.name,
            "status": "historico_atualizado",
            "registros_adicionados": int(len(novos)),
            "registros_totais": int(len(combinado)),
            "quantidade_lojas": int(
                combinado["loja_id"].nunique()
            ),
            "ultimas_datas_por_loja": ultimas,
            "modelo_retreinado": False,
            "avaliacao_externa_preservada": True
        }
