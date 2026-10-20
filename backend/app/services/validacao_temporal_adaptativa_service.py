
"""
Validação temporal adaptativa para datasets da API.

Escolhe quantidade e duração das dobras conforme
o histórico disponível no treinamento externo.

Mantém janelas expansivas e horizonte de sete dias.
"""

import numpy as np
import pandas as pd


HORIZONTE = 7
MIN_DIAS_TREINO_INICIAL = 28


def escolher_protocolo(dias_treino: int) -> tuple[int, int]:
    """
    Retorna (numero_dobras, dias_por_validacao).

    A decisão considera somente o treinamento externo.
    """

    if dias_treino >= 240:
        return 4, 28

    if dias_treino >= 144:
        return 3, 21

    if dias_treino >= 72:
        return 2, 14

    raise ValueError(
        "Treinamento externo insuficiente para "
        "a validação temporal adaptativa."
    )


def criar_divisoes(
    df: pd.DataFrame
) -> tuple[list, list]:
    """
    Mantém a interface da função experimental,
    mas utiliza um protocolo adaptativo.

    Retorna índices posicionais e relatório das dobras.
    """

    if df.empty:
        raise ValueError("Treinamento vazio.")

    obrigatorias = {
        "loja_id",
        "data",
        "data_origem"
    }

    faltando = obrigatorias - set(df.columns)

    if faltando:
        raise ValueError(
            f"Metadados ausentes: {sorted(faltando)}"
        )

    dados = df.copy()

    dados["data"] = pd.to_datetime(dados["data"])
    dados["data_origem"] = pd.to_datetime(
        dados["data_origem"]
    )

    if dados[
        ["loja_id", "data", "data_origem"]
    ].isna().any().any():
        raise ValueError(
            "Existem metadados temporais ausentes."
        )

    if not (
        dados["data"] - dados["data_origem"]
        == pd.Timedelta(days=HORIZONTE)
    ).all():
        raise ValueError(
            "Horizonte de previsão inconsistente."
        )

    if dados.duplicated(["loja_id", "data"]).any():
        raise ValueError(
            "Existem registros duplicados por loja e data."
        )

    dias_treino = int(
        dados["data"].nunique()
    )

    quantidade_dobras, dias_validacao = (
        escolher_protocolo(dias_treino)
    )

    primeira_data = dados["data"].min().normalize()
    ultima_data = dados["data"].max().normalize()

    divisoes = []
    relatorio = []

    for numero in range(quantidade_dobras):
        blocos_restantes = quantidade_dobras - numero

        fim_validacao = (
            ultima_data
            - pd.Timedelta(
                days=(blocos_restantes - 1) * dias_validacao
            )
        )

        inicio_validacao = (
            fim_validacao
            - pd.Timedelta(days=dias_validacao - 1)
        )

        primeira_origem = (
            inicio_validacao
            - pd.Timedelta(days=HORIZONTE)
        )

        # Só usamos alvos conhecidos no momento
        # da primeira previsão dessa validação.
        mascara_treino = (
            (dados["data"] <= primeira_origem)
            & (dados["data_origem"] >= primeira_data)
        )

        mascara_validacao = (
            (dados["data"] >= inicio_validacao)
            & (dados["data"] <= fim_validacao)
        )

        idx_treino = np.flatnonzero(
            mascara_treino.to_numpy()
        )

        idx_validacao = np.flatnonzero(
            mascara_validacao.to_numpy()
        )

        if not len(idx_treino) or not len(idx_validacao):
            raise ValueError(
                f"Dobra {numero + 1} sem observações "
                "suficientes. Verifique as lacunas do CSV."
            )

        treino = dados.iloc[idx_treino]
        validacao = dados.iloc[idx_validacao]

        if (
            treino["data"].max()
            > validacao["data_origem"].min()
        ):
            raise ValueError(
                f"Vazamento temporal na dobra {numero + 1}."
            )

        dias_historico = (
            primeira_origem - primeira_data
        ).days + 1

        if dias_historico < MIN_DIAS_TREINO_INICIAL:
            raise ValueError(
                f"Dobra {numero + 1}: "
                "histórico inicial insuficiente."
            )

        if np.intersect1d(
            idx_treino,
            idx_validacao
        ).size:
            raise ValueError(
                "Treino e validação compartilham registros."
            )

        divisoes.append(
            (idx_treino, idx_validacao)
        )

        relatorio.append({
            "divisao": numero + 1,
            "protocolo": "adaptativo",
            "quantidade_divisoes": quantidade_dobras,
            "dias_por_validacao": dias_validacao,
            "treino_inicio": str(
                treino["data"].min().date()
            ),
            "treino_fim": str(
                treino["data"].max().date()
            ),
            "validacao_inicio": str(
                validacao["data"].min().date()
            ),
            "validacao_fim": str(
                validacao["data"].max().date()
            ),
            "primeira_origem_validacao": str(
                validacao["data_origem"].min().date()
            ),
            "linhas_treino": int(len(idx_treino)),
            "linhas_validacao": int(len(idx_validacao)),
            "lojas_treino": int(
                treino["loja_id"].nunique()
            ),
            "lojas_validacao": int(
                validacao["loja_id"].nunique()
            )
        })

    return divisoes, relatorio
