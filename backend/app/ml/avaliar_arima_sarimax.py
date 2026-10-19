
"""
Avaliação externa de ARIMA, SARIMAX e XGBoost
com origens distribuídas ao longo de 2015.

- 20 lojas;
- até 2 previsões por loja e por mês;
- horizonte de 7 dias;
- ordens estatísticas selecionadas previamente;
- parâmetros estatísticos ajustados até 2015-01-23;
- atualização de estado sem reajuste de parâmetros;
- comparação nas mesmas lojas e datas.

Não altera o modelo global XGBoost.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


RAIZ = Path(__file__).resolve().parents[3]
BRUTOS = RAIZ / "dados" / "brutos"
RESULTADOS = RAIZ / "backend" / "resultados"

DATA_CORTE = pd.Timestamp("2015-01-23")
HORIZONTE = 7
MAX_ITER = 80

LOJAS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 14, 15, 16,
    17, 18, 19, 21, 23
]

PREVISOES_POR_MES = 2

ORDEM_ARIMA = (2, 0, 1)
ORDEM_SARIMAX = (1, 0, 1)
SAZONAL_SARIMAX = (1, 0, 0, 7)

ARQUIVO_XGBOOST = (
    RESULTADOS / "previsoes_teste_externo.csv"
)

ARQUIVO_CSV = (
    RESULTADOS / "comparacao_arima_sarimax_mensal.csv"
)

ARQUIVO_JSON = (
    RESULTADOS / "relatorio_arima_sarimax_mensal.json"
)


def criar_exogenas(datas):
    indice = pd.DatetimeIndex(datas)

    return pd.DataFrame(
        {
            "fim_semana": (
                indice.dayofweek >= 5
            ).astype(float)
        },
        index=indice
    )


def calcular_metricas(reais, previstas):
    reais = np.asarray(reais, dtype=float)
    previstas = np.asarray(previstas, dtype=float)

    erros = reais - previstas

    return {
        "rmse": float(
            np.sqrt(np.mean(erros ** 2))
        ),
        "mae": float(
            np.mean(np.abs(erros))
        )
    }


def carregar_dados():
    print("Carregando previsões externas do XGBoost...")

    previsoes = pd.read_csv(
        ARQUIVO_XGBOOST,
        parse_dates=["data", "data_origem"]
    )

    previsoes = previsoes.loc[
        previsoes["loja_id"].isin(LOJAS)
    ].copy()

    if previsoes.empty:
        raise ValueError("Não existem previsões externas.")

    if previsoes.duplicated(
        ["loja_id", "data"]
    ).any():
        raise ValueError(
            "Existem previsões duplicadas."
        )

    diferenca = (
        previsoes["data"]
        - previsoes["data_origem"]
    ).dt.days

    if not (diferenca == HORIZONTE).all():
        raise ValueError(
            "Horizonte de previsão inconsistente."
        )

    if (
        previsoes["data_origem"] < DATA_CORTE
    ).any():
        raise ValueError(
            "Origem anterior ao corte temporal."
        )

    print("Carregando histórico original...")

    bruto = pd.read_csv(
        BRUTOS / "train.csv",
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"]
    )

    bruto = bruto.loc[
        bruto["Store"].isin(LOJAS)
    ].copy()

    if bruto.duplicated(
        ["Store", "Date"]
    ).any():
        raise ValueError(
            "Histórico com datas duplicadas por loja."
        )

    return previsoes, bruto


def selecionar_previsoes_mensais(previsoes_loja):
    """
    Seleciona até duas observações distribuídas
    dentro de cada mês, por posição cronológica.
    """
    df = previsoes_loja.sort_values(
        ["data", "data_origem"]
    ).copy()

    df["mes_referencia"] = (
        df["data"].dt.to_period("M")
    )

    selecionadas = []

    for _, grupo in df.groupby(
        "mes_referencia",
        sort=True
    ):
        grupo = grupo.sort_values("data")

        quantidade = min(
            PREVISOES_POR_MES,
            len(grupo)
        )

        if quantidade == 1:
            posicoes = [len(grupo) // 2]
        else:
            posicoes = [
                len(grupo) // 3,
                (2 * len(grupo)) // 3
            ]

        parte = grupo.iloc[
            sorted(set(posicoes))
        ].copy()

        selecionadas.append(parte)

    if not selecionadas:
        raise ValueError(
            "Nenhuma previsão mensal selecionada."
        )

    resultado = pd.concat(
        selecionadas,
        ignore_index=True
    )

    return resultado.sort_values(
        ["data_origem", "data"]
    ).reset_index(drop=True)


def obter_serie_loja(bruto, loja, ultima_origem):
    dados = bruto.loc[
        bruto["Store"] == loja,
        ["Date", "Sales", "Open"]
    ].copy()

    if dados.empty:
        raise ValueError(
            f"Histórico ausente para loja {loja}."
        )

    dados = (
        dados.set_index("Date")
        .sort_index()
    )

    # Dia fechado registrado: venda observada zero.
    dados.loc[
        dados["Open"] == 0,
        "Sales"
    ] = 0.0

    calendario = pd.date_range(
        dados.index.min(),
        ultima_origem,
        freq="D"
    )

    # Datas realmente ausentes permanecem NaN.
    serie = (
        dados["Sales"]
        .reindex(calendario)
        .astype(float)
    )

    return serie


def ajustar_modelo(historico, algoritmo):
    if historico.notna().sum() < 100:
        raise ValueError(
            "Histórico insuficiente para ajuste."
        )

    if algoritmo == "arima":
        ordem = ORDEM_ARIMA
        sazonal = (0, 0, 0, 0)
        exogenas = None
    else:
        ordem = ORDEM_SARIMAX
        sazonal = SAZONAL_SARIMAX
        exogenas = criar_exogenas(
            historico.index
        )

    inicio = time.perf_counter()

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")

        modelo = SARIMAX(
            historico,
            exog=exogenas,
            order=ordem,
            seasonal_order=sazonal,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(
            disp=False,
            maxiter=MAX_ITER
        )

    duracao = time.perf_counter() - inicio

    if not modelo.mle_retvals.get(
        "converged", False
    ):
        raise RuntimeError(
            f"{algoritmo.upper()} não convergiu."
        )

    return modelo, duracao, len(avisos)


def prever_loja(loja, bruto, previsoes_loja):
    selecionadas = selecionar_previsoes_mensais(
        previsoes_loja
    )

    ultima_data = (
        selecionadas["data_origem"].max()
    )

    serie = obter_serie_loja(
        bruto,
        loja,
        ultima_data
    )

    historico = serie.loc[:DATA_CORTE]

    print("  Ajustando ARIMA...", flush=True)

    arima, tempo_arima, avisos_arima = ajustar_modelo(
        historico,
        "arima"
    )

    print("  Ajustando SARIMAX...", flush=True)

    sarimax, tempo_sarimax, avisos_sarimax = (
        ajustar_modelo(
            historico,
            "sarimax"
        )
    )

    ultima_origem = DATA_CORTE
    resultados = []

    for _, linha in selecionadas.iterrows():
        origem = linha["data_origem"]
        data_alvo = linha["data"]

        if (data_alvo - origem).days != HORIZONTE:
            raise ValueError(
                "Horizonte de previsão inconsistente."
            )

        if origem < ultima_origem:
            raise ValueError(
                "Origens fora da ordem cronológica."
            )

        # Atualiza estado exclusivamente com vendas
        # observadas até a origem atual.
        if origem > ultima_origem:
            novas_datas = pd.date_range(
                ultima_origem + pd.Timedelta(days=1),
                origem,
                freq="D"
            )

            novos_valores = serie.reindex(
                novas_datas
            )

            arima = arima.append(
                novos_valores,
                refit=False
            )

            sarimax = sarimax.append(
                novos_valores,
                exog=criar_exogenas(novas_datas),
                refit=False
            )

        datas_futuras = pd.date_range(
            origem + pd.Timedelta(days=1),
            periods=HORIZONTE,
            freq="D"
        )

        previsao_arima = float(
            arima.get_forecast(
                steps=HORIZONTE
            ).predicted_mean.iloc[-1]
        )

        previsao_sarimax = float(
            sarimax.get_forecast(
                steps=HORIZONTE,
                exog=criar_exogenas(datas_futuras)
            ).predicted_mean.iloc[-1]
        )

        resultados.append({
            "loja_id": int(loja),
            "data": data_alvo,
            "data_origem": origem,
            "mes": data_alvo.strftime("%Y-%m"),
            "vendas_reais": float(
                linha["vendas_reais"]
            ),
            "xgboost": float(
                linha["vendas_previstas"]
            ),
            "arima": previsao_arima,
            "sarimax": previsao_sarimax
        })

        ultima_origem = origem

    diagnostico = {
        "loja_id": int(loja),
        "previsoes": len(resultados),
        "tempo_ajuste_arima": tempo_arima,
        "tempo_ajuste_sarimax": tempo_sarimax,
        "avisos_arima": avisos_arima,
        "avisos_sarimax": avisos_sarimax,
    }

    return resultados, diagnostico


def main():
    RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("========================================")
    print("COMPARAÇÃO ESTATÍSTICA MENSAL")
    print("========================================")

    inicio = time.perf_counter()

    previsoes, bruto = carregar_dados()

    todos = []
    diagnosticos = []

    for indice, loja in enumerate(
        LOJAS, start=1
    ):
        print(
            f"\nLoja {loja} ({indice}/{len(LOJAS)})",
            flush=True
        )

        parte = previsoes.loc[
            previsoes["loja_id"] == loja
        ]

        if parte.empty:
            raise ValueError(
                f"Não há previsões para loja {loja}."
            )

        resultados, diagnostico = prever_loja(
            loja, bruto, parte
        )

        todos.extend(resultados)
        diagnosticos.append(diagnostico)

        print(
            f"  Previsões: {len(resultados)}",
            flush=True
        )

    df = pd.DataFrame(todos)

    if df.empty:
        raise ValueError(
            "Nenhuma previsão foi produzida."
        )

    if df.duplicated(
        ["loja_id", "data"]
    ).any():
        raise ValueError(
            "Existem previsões duplicadas."
        )

    for coluna in [
        "vendas_reais",
        "xgboost",
        "arima",
        "sarimax"
    ]:
        if not np.isfinite(
            df[coluna].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Valores inválidos em {coluna}."
            )

    metricas = {}

    for modelo in [
        "xgboost", "arima", "sarimax"
    ]:
        metricas[modelo] = calcular_metricas(
            df["vendas_reais"],
            df[modelo]
        )

    metricas_mensais = {}

    for mes, grupo in df.groupby(
        "mes", sort=True
    ):
        metricas_mensais[mes] = {
            "observacoes": len(grupo),
            "modelos": {
                modelo: calcular_metricas(
                    grupo["vendas_reais"],
                    grupo[modelo]
                )
                for modelo in [
                    "xgboost", "arima", "sarimax"
                ]
            }
        }

    df.to_csv(
        ARQUIVO_CSV,
        index=False,
        date_format="%Y-%m-%d"
    )

    duracao = time.perf_counter() - inicio

    relatorio = {
        "tipo": "comparacao_externa_mensal_amostral",
        "lojas": LOJAS,
        "quantidade_lojas": len(LOJAS),
        "previsoes_por_mes": PREVISOES_POR_MES,
        "quantidade_previsoes": len(df),
        "horizonte_dias": HORIZONTE,
        "data_corte_treinamento": str(
            DATA_CORTE.date()
        ),
        "ordem_arima": ORDEM_ARIMA,
        "ordem_sarimax": ORDEM_SARIMAX,
        "ordem_sazonal": SAZONAL_SARIMAX,
        "parametros_reajustados_no_teste": False,
        "metricas_gerais": metricas,
        "metricas_mensais": metricas_mensais,
        "diagnosticos": diagnosticos,
        "tempo_total_segundos": round(
            duracao, 3
        ),
        "observacao": (
            "Comparação pareada em datas distribuídas "
            "ao longo dos meses do teste externo. "
            "Os modelos estatísticos foram ajustados "
            "até 2015-01-23 e seus estados foram "
            "atualizados com observações disponíveis "
            "até cada origem, sem refit."
        )
    }

    ARQUIVO_JSON.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========================================")
    print("RESULTADOS GERAIS")
    print("========================================")

    print(f"Lojas: {len(LOJAS)}")
    print(f"Previsões: {len(df)}")

    for modelo, valores in metricas.items():
        print(
            f"{modelo.upper()}: "
            f"RMSE={valores['rmse']:.2f} | "
            f"MAE={valores['mae']:.2f}"
        )

    print("\n=== RMSE POR MÊS ===")

    for mes, dados in metricas_mensais.items():
        print(
            f"{mes}: "
            f"XGBoost="
            f"{dados['modelos']['xgboost']['rmse']:.2f} | "
            f"ARIMA="
            f"{dados['modelos']['arima']['rmse']:.2f} | "
            f"SARIMAX="
            f"{dados['modelos']['sarimax']['rmse']:.2f}"
        )

    print(
        f"\nTempo total: {duracao:.2f}s"
    )

    print("\nArquivos gerados:")
    print(ARQUIVO_CSV)
    print(ARQUIVO_JSON)


if __name__ == "__main__":
    main()
