
"""
Divisão temporal dos datasets enviados à ferramenta.

Reserva aproximadamente 20% das datas finais para
teste externo, respeitando horizonte de sete dias.

O teste nunca participa da seleção do modelo.
"""

import pandas as pd


HORIZONTE = 7
PROPORCAO_TREINO = 0.80
MIN_DIAS_DISTINTOS = 90


def dividir_dataset_temporalmente(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    if df.empty:
        raise ValueError("Dataset vazio.")

    dados = df.copy()

    dados["data"] = pd.to_datetime(
        dados["data"],
        errors="raise"
    )

    dados["data_origem"] = pd.to_datetime(
        dados["data_origem"],
        errors="raise"
    )

    if dados[
        ["data", "data_origem"]
    ].isna().any().any():
        raise ValueError(
            "Datas de previsão ausentes."
        )

    if not (
        dados["data"] - dados["data_origem"]
        == pd.Timedelta(days=HORIZONTE)
    ).all():
        raise ValueError(
            "Horizonte temporal inconsistente."
        )

    datas = pd.DatetimeIndex(
        sorted(dados["data"].unique())
    )

    if len(datas) < MIN_DIAS_DISTINTOS:
        raise ValueError(
            "São necessários pelo menos "
            f"{MIN_DIAS_DISTINTOS} dias distintos."
        )

    indice_corte = (
        int(len(datas) * PROPORCAO_TREINO) - 1
    )

    data_corte = datas[indice_corte]

    # Só utilizar vendas conhecidas até o corte.
    treino = dados.loc[
        dados["data"] <= data_corte
    ].copy()

    # Cada previsão de teste deve ter origem
    # no corte ou depois dele.
    teste = dados.loc[
        dados["data_origem"] >= data_corte
    ].copy()

    if treino.empty or teste.empty:
        raise ValueError(
            "Treinamento ou teste externo vazio."
        )

    if treino["data"].max() >= teste["data"].min():
        raise ValueError(
            "Sobreposição temporal entre treino e teste."
        )

    if teste["data_origem"].min() < data_corte:
        raise ValueError(
            "O teste utiliza origem anterior ao corte."
        )

    relatorio = {
        "data_corte": str(data_corte.date()),
        "horizonte_dias": HORIZONTE,
        "proporcao_treino": PROPORCAO_TREINO,
        "linhas_treino": int(len(treino)),
        "linhas_teste": int(len(teste)),
        "dias_distintos_total": int(len(datas)),
        "dias_distintos_treino": int(
            treino["data"].nunique()
        ),
        "dias_distintos_teste": int(
            teste["data"].nunique()
        ),
        "lojas_treino": int(
            treino["loja_id"].nunique()
        ),
        "lojas_teste": int(
            teste["loja_id"].nunique()
        ),
        "primeira_data_teste": str(
            teste["data"].min().date()
        ),
        "primeira_origem_teste": str(
            teste["data_origem"].min().date()
        )
    }

    return treino, teste, relatorio
