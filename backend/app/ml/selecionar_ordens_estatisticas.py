
"""
Seleção preliminar de ordens ARIMA e SARIMAX.

Utiliza quatro dobras temporais do treinamento,
sem acessar o conjunto externo de teste.

Para cada configuração candidata:
- Ajusta os modelos com dados disponíveis na origem;
- Atualiza o estado com vendas observadas;
- Faz previsões com horizonte de 7 dias;
- Calcula RMSE e MAE agregados.

Este é um teste piloto com duas lojas.
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
TREINAMENTO = RAIZ / "dados" / "processados" / "treinamento"
RESULTADOS = RAIZ / "backend" / "resultados"

LOJAS = [
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10,
    11, 12, 14, 15, 16,
    17, 18, 19, 21, 23
]
N_DOBRAS = 4
PREVISOES_POR_LOJA_DOBRA = 7
HORIZONTE = 7
MAX_ITER = 70

CANDIDATOS = {
    "arima": [
        {
            "order": (1, 0, 1),
            "seasonal_order": (0, 0, 0, 0),
        },
        {
            "order": (2, 0, 1),
            "seasonal_order": (0, 0, 0, 0),
        },
    ],
    "sarimax": [
        {
            "order": (1, 0, 1),
            "seasonal_order": (1, 0, 0, 7),
        },
        {
            "order": (1, 0, 1),
            "seasonal_order": (0, 0, 1, 7),
        },
    ],
}


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


def carregar_dados():
    print("Carregando dados originais...")

    bruto = pd.read_csv(
        BRUTOS / "train.csv",
        usecols=["Store", "Date", "Sales", "Open"],
        parse_dates=["Date"]
    )

    if bruto.duplicated(["Store", "Date"]).any():
        raise ValueError("Existem registros duplicados.")

    bruto = bruto[
        bruto["Store"].isin(LOJAS)
    ].copy()

    bruto.loc[
        bruto["Open"] == 0, "Sales"
    ] = 0.0

    print("Carregando metadados e dobras temporais...")

    metadata = pd.read_csv(
        TREINAMENTO / "metadata_treino.csv",
        parse_dates=["data", "data_origem"]
    )

    divisoes = []

    with np.load(
        TREINAMENTO / "indices_validacao_temporal.npz"
    ) as indices:
        for numero in range(1, N_DOBRAS + 1):
            idx_treino = indices[
                f"treino_{numero}"
            ]

            idx_validacao = indices[
                f"validacao_{numero}"
            ]

            treino = metadata.iloc[idx_treino]
            validacao = metadata.iloc[idx_validacao]

            validacao = validacao[
                validacao["loja_id"].isin(LOJAS)
            ].copy()

            # Selecionar as primeiras previsões
            # disponíveis de cada loja nesta dobra.
            validacao = (
                validacao
                .sort_values(["loja_id", "data"])
                .groupby("loja_id", group_keys=False)
                .head(PREVISOES_POR_LOJA_DOBRA)
            )

            corte_ajuste = treino["data"].max()

            if validacao.empty:
                raise ValueError(
                    f"Dobra {numero} sem validação."
                )

            if (
                validacao["data_origem"].min()
                < corte_ajuste
            ):
                raise ValueError(
                    f"Inconsistência temporal na dobra {numero}."
                )

            divisoes.append({
                "numero": numero,
                "validacao": validacao,
                "corte_ajuste": corte_ajuste,
            })

    return bruto, divisoes


def obter_serie(bruto, loja):
    dados = bruto[
        bruto["Store"] == loja
    ].set_index("Date").sort_index()

    calendario = pd.date_range(
        dados.index.min(),
        dados.index.max(),
        freq="D"
    )

    # Dias ausentes continuam NaN.
    # Dias fechados presentes no arquivo são zero.
    return (
        dados["Sales"]
        .reindex(calendario)
        .astype(float)
    )


def ajustar(serie, config, usar_exogenas):
    exogenas = (
        criar_exogenas(serie.index)
        if usar_exogenas
        else None
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        modelo = SARIMAX(
            serie,
            exog=exogenas,
            order=tuple(config["order"]),
            seasonal_order=tuple(
                config["seasonal_order"]
            ),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        resultado = modelo.fit(
            disp=False,
            maxiter=MAX_ITER
        )

    if not resultado.mle_retvals.get(
        "converged", False
    ):
        raise RuntimeError("Modelo não convergiu.")

    return resultado


def prever_origens(
    serie,
    validacao,
    config,
    usar_exogenas
):
    validacao = validacao.sort_values(
        "data_origem"
    )

    primeira_origem = (
        validacao["data_origem"].min()
    )

    historico = serie.loc[:primeira_origem]

    if historico.notna().sum() < 100:
        raise ValueError(
            "Histórico insuficiente para ajuste."
        )

    resultado = ajustar(
        historico,
        config,
        usar_exogenas
    )

    ultima_origem = primeira_origem

    previsoes = []

    for _, linha in validacao.iterrows():
        origem = linha["data_origem"]
        alvo = linha["data"]

        if (alvo - origem).days != HORIZONTE:
            raise ValueError(
                "Horizonte de previsão inconsistente."
            )

        # Atualizar somente com valores disponíveis
        # até a nova origem, sem reajustar parâmetros.
        if origem > ultima_origem:
            novas_datas = pd.date_range(
                ultima_origem + pd.Timedelta(days=1),
                origem,
                freq="D"
            )

            novos_valores = serie.reindex(
                novas_datas
            )

            novas_exogenas = (
                criar_exogenas(novas_datas)
                if usar_exogenas
                else None
            )

            resultado = resultado.append(
                novos_valores,
                exog=novas_exogenas,
                refit=False
            )

        futuras_datas = pd.date_range(
            origem + pd.Timedelta(days=1),
            periods=HORIZONTE,
            freq="D"
        )

        futuras_exogenas = (
            criar_exogenas(futuras_datas)
            if usar_exogenas
            else None
        )

        previsao = float(
            resultado.get_forecast(
                steps=HORIZONTE,
                exog=futuras_exogenas
            ).predicted_mean.iloc[-1]
        )

        if not np.isfinite(previsao):
            raise ValueError(
                "Previsão não finita."
            )

        real = serie.loc[alvo]

        if not np.isfinite(real):
            raise ValueError(
                "Venda real não disponível."
            )

        previsoes.append({
            "loja_id": int(linha["loja_id"]),
            "data": str(alvo.date()),
            "real": float(real),
            "previsto": previsao
        })

        ultima_origem = origem

    return previsoes


def avaliar_candidato(
    bruto,
    divisoes,
    algoritmo,
    config
):
    usar_exogenas = algoritmo == "sarimax"

    todas_previsoes = []
    inicio = time.perf_counter()

    for dobra in divisoes:
        numero = dobra["numero"]
        validacao = dobra["validacao"]

        for loja in LOJAS:
            validacao_loja = validacao[
                validacao["loja_id"] == loja
            ]

            if validacao_loja.empty:
                raise ValueError(
                    f"Loja {loja} ausente da dobra {numero}."
                )

            serie = obter_serie(bruto, loja)

            resultado = prever_origens(
                serie,
                validacao_loja,
                config,
                usar_exogenas
            )

            todas_previsoes.extend(resultado)

    reais = np.array([
        item["real"]
        for item in todas_previsoes
    ])

    previstas = np.array([
        item["previsto"]
        for item in todas_previsoes
    ])

    erros = reais - previstas

    return {
        "rmse": float(
            np.sqrt(np.mean(erros ** 2))
        ),
        "mae": float(
            np.mean(np.abs(erros))
        ),
        "observacoes": len(todas_previsoes),
        "tempo_segundos": round(
            time.perf_counter() - inicio,
            3
        )
    }


def main():
    RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    print("========================================")
    print("SELEÇÃO PILOTO DE ORDENS ESTATÍSTICAS")
    print("========================================")

    bruto, divisoes = carregar_dados()

    esperado = sum(
        len(dobra["validacao"])
        for dobra in divisoes
    )

    print(f"Lojas: {LOJAS}")
    print(f"Dobras: {N_DOBRAS}")
    print(f"Observações esperadas: {esperado}")

    resultados = {
        "tipo": "selecao_preliminar",
        "horizonte": HORIZONTE,
        "lojas": LOJAS,
        "dobras": N_DOBRAS,
        "observacoes_esperadas": esperado,
        "modelos": {}
    }

    for algoritmo, candidatos in CANDIDATOS.items():
        print(f"\n=== {algoritmo.upper()} ===")

        avaliacoes = []

        for indice, config in enumerate(
            candidatos, start=1
        ):
            print(
                f"Candidato {indice}: {config}",
                flush=True
            )

            try:
                metricas = avaliar_candidato(
                    bruto,
                    divisoes,
                    algoritmo,
                    config
                )

                registro = {
                    "configuracao": config,
                    "status": "sucesso",
                    **metricas
                }

                print(
                    f"RMSE={metricas['rmse']:.2f} | "
                    f"MAE={metricas['mae']:.2f} | "
                    f"Tempo={metricas['tempo_segundos']:.2f}s",
                    flush=True
                )

            except Exception as erro:
                registro = {
                    "configuracao": config,
                    "status": "falha",
                    "erro": str(erro)
                }

                print(
                    f"Falha: {erro}",
                    flush=True
                )

            avaliacoes.append(registro)

        validos = [
            item for item in avaliacoes
            if item["status"] == "sucesso"
            and item["observacoes"] == esperado
        ]

        melhor = (
            min(validos, key=lambda x: x["rmse"])
            if validos
            else None
        )

        resultados["modelos"][algoritmo] = {
            "candidatos": avaliacoes,
            "melhor_configuracao": (
                melhor["configuracao"]
                if melhor else None
            ),
            "melhor_rmse": (
                melhor["rmse"]
                if melhor else None
            )
        }

    destino = (
        RESULTADOS
        / "selecao_ordens_estatisticas_piloto.json"
    )

    destino.write_text(
        json.dumps(
            resultados,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n========================================")
    print("RESUMO DA SELEÇÃO")
    print("========================================")

    for algoritmo, resultado in (
        resultados["modelos"].items()
    ):
        print(f"\n{algoritmo.upper()}")
        print(
            "Melhor configuração: "
            f"{resultado['melhor_configuracao']}"
        )
        print(
            "RMSE: "
            f"{resultado['melhor_rmse']}"
        )

    print(f"\nRelatório: {destino}")
    print(
        "Seleção preliminar concluída. "
        "Nenhum dado externo de teste foi utilizado."
    )


if __name__ == "__main__":
    main()
