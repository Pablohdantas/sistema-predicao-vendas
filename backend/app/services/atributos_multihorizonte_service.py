
"""
Engenharia de atributos para previsão dos dias 1 a 7.

Cada linha representa uma previsão feita em uma
data de origem para uma data-alvo posterior.

Somente vendas conhecidas até a origem podem
ser utilizadas como atributos.

Dias ausentes permanecem NaN, não zero.
"""

import numpy as np
import pandas as pd


HORIZONTES = range(1, 8)

ATRIBUTOS_HISTORICOS = [
    "lag_origem",
    "lag_7_origem",
    "lag_14_origem",
    "lag_21_origem",
    "lag_28_origem",
    "media_7_origem",
    "media_28_origem",
    "observacoes_7_origem",
    "observacoes_28_origem",
]

ATRIBUTOS_ESTATICOS = [
    "tipo_loja",
    "sortimento",
    "distancia_concorrente",
]


def _preparar_base_loja(
    dados: pd.DataFrame,
    loja_id: int,
) -> pd.DataFrame:
    """
    Produz atributos disponíveis em cada data de
    origem, sem consultar observações posteriores.
    """

    obrigatorias = {"loja_id", "data", "vendas"}

    faltantes = obrigatorias - set(dados.columns)

    if faltantes:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {sorted(faltantes)}"
        )

    df = dados.copy()

    df["data"] = pd.to_datetime(
        df["data"],
        format="%Y-%m-%d",
        errors="raise",
    )

    df["vendas"] = pd.to_numeric(
        df["vendas"],
        errors="raise",
    )

    ids = pd.to_numeric(
        df["loja_id"],
        errors="coerce",
    )

    loja = df.loc[ids == loja_id].copy()

    if loja.empty:
        raise LookupError(
            f"Loja {loja_id} não encontrada."
        )

    if loja["data"].duplicated().any():
        raise ValueError(
            "Há datas duplicadas para a loja."
        )

    loja = loja.sort_values("data")

    calendario = pd.date_range(
        start=loja["data"].min(),
        end=loja["data"].max(),
        freq="D",
    )

    base = (
        loja.set_index("data")
        .reindex(calendario)
    )

    base.index.name = "data"

    vendas = base["vendas"]

    # A venda da origem já é conhecida no instante
    # em que uma previsão futura será gerada.
    base["lag_origem"] = vendas

    for atraso in (7, 14, 21, 28):
        base[f"lag_{atraso}_origem"] = (
            vendas.shift(atraso)
        )

    # As janelas terminam na data de origem.
    # Dias sem registro permanecem desconhecidos.
    base["media_7_origem"] = (
        vendas.rolling(
            window=7,
            min_periods=4,
        ).mean()
    )

    base["media_28_origem"] = (
        vendas.rolling(
            window=28,
            min_periods=14,
        ).mean()
    )

    base["observacoes_7_origem"] = (
        vendas.rolling(
            window=7,
            min_periods=1,
        ).count()
    )

    base["observacoes_28_origem"] = (
        vendas.rolling(
            window=28,
            min_periods=1,
        ).count()
    )

    base["loja_id"] = loja_id

    return base


def _adicionar_calendario_alvo(
    dados: pd.DataFrame,
) -> pd.DataFrame:
    """Cria atributos conhecidos da data prevista."""

    resultado = dados.copy()

    datas = pd.to_datetime(resultado["data"])

    resultado["dia_semana"] = datas.dt.dayofweek
    resultado["dia_mes"] = datas.dt.day
    resultado["mes"] = datas.dt.month

    resultado["semana_ano"] = (
        datas.dt.isocalendar().week.astype(int)
    )

    return resultado


def gerar_exemplos_treinamento_loja(
    dados: pd.DataFrame,
    loja_id: int,
) -> pd.DataFrame:
    """
    Gera exemplos supervisionados dos horizontes 1 a 7.

    A variável-alvo é a venda real da data futura.
    Os atributos históricos pertencem à data de origem.

    Use a divisão temporal antes de ajustar o modelo.
    """

    base = _preparar_base_loja(dados, loja_id)

    vendas = base["vendas"]

    colunas_origem = [
        "loja_id",
        *ATRIBUTOS_HISTORICOS,
        *[
            coluna
            for coluna in ATRIBUTOS_ESTATICOS
            if coluna in base.columns
        ],
    ]

    partes = []

    for horizonte in HORIZONTES:
        exemplos = base[colunas_origem].copy()

        exemplos["data_origem"] = base.index

        exemplos["data"] = (
            base.index + pd.Timedelta(days=horizonte)
        )

        exemplos["horizonte_dias"] = horizonte

        # Valor-alvo futuro: somente para treinamento.
        # Ele nunca é utilizado nos atributos históricos.
        exemplos["vendas"] = vendas.shift(
            -horizonte
        ).to_numpy()

        # A origem precisa ter uma venda observada.
        # O alvo também precisa ser conhecido.
        validas = (
            base["vendas"].notna().to_numpy()
            & exemplos["vendas"].notna().to_numpy()
        )

        exemplos = exemplos.loc[validas].copy()

        partes.append(
            _adicionar_calendario_alvo(
                exemplos.reset_index(drop=True)
            )
        )

    resultado = pd.concat(
        partes,
        ignore_index=True,
    )

    resultado = resultado.sort_values(
        ["data_origem", "horizonte_dias"]
    ).reset_index(drop=True)

    return resultado


def gerar_entradas_futuras_loja(
    dados: pd.DataFrame,
    loja_id: int,
) -> pd.DataFrame:
    """
    Cria as sete entradas futuras a partir da
    última observação disponível da loja.

    Não inventa vendas futuras.
    """

    base = _preparar_base_loja(dados, loja_id)

    observadas = base.loc[
        base["vendas"].notna()
    ]

    if observadas.empty:
        raise ValueError(
            "A loja não possui vendas observadas."
        )

    origem = observadas.index.max()
    ultima = observadas.loc[origem]

    colunas = [
        "loja_id",
        *ATRIBUTOS_HISTORICOS,
        *[
            coluna
            for coluna in ATRIBUTOS_ESTATICOS
            if coluna in base.columns
        ],
    ]

    entradas = []

    for horizonte in HORIZONTES:
        linha = {
            coluna: ultima[coluna]
            for coluna in colunas
        }

        linha["data_origem"] = origem
        linha["data"] = (
            origem + pd.Timedelta(days=horizonte)
        )
        linha["horizonte_dias"] = horizonte

        # Na previsão, o alvo é desconhecido.
        linha["vendas"] = np.nan

        entradas.append(linha)

    return _adicionar_calendario_alvo(
        pd.DataFrame(entradas)
    )
