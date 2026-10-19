
"""
Validação temporal para previsão de vendas com horizonte de 7 dias.

Cria 4 divisões temporais expansivas usando apenas os dados
de treinamento. Garante que os alvos usados no ajuste já
eram conhecidos na primeira origem de cada validação.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


DADOS = Path(__file__).resolve().parents[1]
PASTA = DADOS / "processados" / "treinamento"

HORIZONTE = 7
N_DIVISOES = 4
DIAS_VALIDACAO = 28
MIN_DIAS_TREINO = 180


def carregar_metadados():
    caminho = PASTA / "metadata_treino.csv"

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    df = pd.read_csv(
        caminho,
        parse_dates=["data", "data_origem"]
    )

    if df.empty:
        raise ValueError("Metadados de treinamento vazios.")

    if df[["loja_id", "data", "data_origem"]].isna().any().any():
        raise ValueError("Existem metadados obrigatórios ausentes.")

    if not (
        df["data"] - df["data_origem"]
        == pd.Timedelta(days=HORIZONTE)
    ).all():
        raise ValueError("Horizonte temporal inconsistente.")

    if df.duplicated(["loja_id", "data"]).any():
        raise ValueError("Existem registros duplicados.")

    return df


def criar_divisoes(df):
    ultima_data = df["data"].max().normalize()
    primeira_data = df["data"].min().normalize()

    divisoes = []
    relatorio = []

    # Quatro janelas de validação de 28 dias,
    # posicionadas no final do período de treinamento.
    for numero in range(N_DIVISOES):
        blocos_restantes = N_DIVISOES - numero

        fim_validacao = (
            ultima_data
            - pd.Timedelta(
                days=(blocos_restantes - 1) * DIAS_VALIDACAO
            )
        )

        inicio_validacao = (
            fim_validacao
            - pd.Timedelta(days=DIAS_VALIDACAO - 1)
        )

        # Primeira previsão produzida no início
        # do bloco de validação.
        primeira_origem = (
            inicio_validacao - pd.Timedelta(days=HORIZONTE)
        )

        # Só usar alvos conhecidos até essa origem.
        mascara_treino = (
            df["data"] <= primeira_origem
        ) & (
            df["data_origem"] >= primeira_data
        )

        mascara_validacao = (
            (df["data"] >= inicio_validacao)
            & (df["data"] <= fim_validacao)
        )

        indices_treino = np.flatnonzero(
            mascara_treino.to_numpy()
        )

        indices_validacao = np.flatnonzero(
            mascara_validacao.to_numpy()
        )

        if not len(indices_treino) or not len(indices_validacao):
            raise ValueError(
                f"Divisão {numero + 1} sem dados suficientes."
            )

        treino = df.iloc[indices_treino]
        validacao = df.iloc[indices_validacao]

        if (
            treino["data"].max()
            > validacao["data_origem"].min()
        ):
            raise AssertionError(
                "Vazamento temporal identificado."
            )

        dias_historico = (
            primeira_origem - primeira_data
        ).days + 1

        if dias_historico < MIN_DIAS_TREINO:
            raise ValueError(
                "Histórico insuficiente para validação."
            )

        if np.intersect1d(
            indices_treino, indices_validacao
        ).size:
            raise AssertionError(
                "Treino e validação compartilham linhas."
            )

        divisoes.append(
            (indices_treino, indices_validacao)
        )

        relatorio.append({
            "divisao": numero + 1,
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
            "linhas_treino": len(indices_treino),
            "linhas_validacao": len(indices_validacao),
            "lojas_treino": int(
                treino["loja_id"].nunique()
            ),
            "lojas_validacao": int(
                validacao["loja_id"].nunique()
            )
        })

    return divisoes, relatorio


def main():
    df = carregar_metadados()

    divisoes, relatorio = criar_divisoes(df)

    # Salvar os índices sem modificar X_treino ou y_treino.
    arrays = {}

    for i, (treino, validacao) in enumerate(
        divisoes, start=1
    ):
        arrays[f"treino_{i}"] = treino
        arrays[f"validacao_{i}"] = validacao

    np.savez_compressed(
        PASTA / "indices_validacao_temporal.npz",
        **arrays
    )

    resultado = {
        "horizonte": HORIZONTE,
        "quantidade_divisoes": N_DIVISOES,
        "dias_por_validacao": DIAS_VALIDACAO,
        "tipo": "janelas expansivas",
        "divisoes": relatorio
    }

    (PASTA / "relatorio_validacao_temporal.json").write_text(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False
        )
    )

    print("\nValidação temporal preparada com sucesso!")


if __name__ == "__main__":
    main()
