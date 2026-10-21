
"""
Pipeline experimental multihorizonte (dias 1 a 7).

Executa:
1. Engenharia de atributos por loja.
2. Separação temporal externa.
3. Competição LightGBM × XGBoost.
4. Seleção por validação temporal.
5. Treinamento final.
6. Avaliação externa por horizonte.
7. Previsões futuras e total semanal.
8. Persistência independente dos modelos antigos.

Esta versão compara configurações fixas.
A otimização FLAML será integrada posteriormente.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.services.atributos_multihorizonte_service import (
    gerar_exemplos_treinamento_loja,
    gerar_entradas_futuras_loja,
)
from backend.app.services.divisao_multihorizonte_service import (
    dividir_dataset_multihorizonte,
)
from backend.app.services.validacao_multihorizonte_service import (
    criar_dobras_multihorizonte,
)
from backend.app.services.preprocessamento_service import (
    ajustar_preprocessamento,
    transformar_atributos,
)
from backend.app.services.automl_service import criar_modelo
from backend.app.services.treinamento_service import pasta_da_execucao


CONFIGURACOES = {
    "lgbm": {
        "n_estimators": 120,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "num_leaves": 31,
    },
    "xgboost": {
        "n_estimators": 120,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
}


def metricas(y_real, y_previsto):
    real = np.asarray(y_real, dtype=float)
    previsto = np.asarray(y_previsto, dtype=float)

    if (
        real.size == 0
        or real.size != previsto.size
        or not np.isfinite(real).all()
        or not np.isfinite(previsto).all()
    ):
        raise ValueError("Valores inválidos nas métricas.")

    erros = real - previsto

    return {
        "rmse": round(
            float(np.sqrt(np.mean(erros ** 2))), 4
        ),
        "mae": round(
            float(np.mean(np.abs(erros))), 4
        ),
        "quantidade": int(len(real)),
    }


def preparar_exemplos(dados):
    partes = []

    for loja_id in dados["loja_id"].dropna().unique():
        parte = gerar_exemplos_treinamento_loja(
            dados,
            int(loja_id),
        )
        partes.append(parte)

    if not partes:
        raise ValueError("Nenhuma loja encontrada.")

    return pd.concat(
        partes,
        ignore_index=True,
    )


def competir_modelos(treino_externo, divisoes):
    resultados = {}

    for algoritmo, configuracao in CONFIGURACOES.items():
        print(f"\nAvaliando {algoritmo}...", flush=True)

        reais = []
        previstas = []
        relatorios = []

        for numero, (idx_treino, idx_validacao) in enumerate(
            divisoes,
            start=1,
        ):
            treino = treino_externo.iloc[idx_treino]
            validacao = treino_externo.iloc[idx_validacao]

            esquema = ajustar_preprocessamento(treino)

            x_treino = transformar_atributos(
                treino,
                esquema,
            )
            x_validacao = transformar_atributos(
                validacao,
                esquema,
            )

            modelo = criar_modelo(
                algoritmo,
                configuracao,
            )

            modelo.fit(
                x_treino,
                treino["vendas"].to_numpy(dtype=float),
            )

            # Vendas previstas não devem ser negativas.
            predicoes = np.maximum(
                0,
                np.asarray(
                    modelo.predict(x_validacao),
                    dtype=float,
                ),
            )

            y_real = validacao["vendas"].to_numpy(
                dtype=float
            )

            reais.extend(y_real.tolist())
            previstas.extend(predicoes.tolist())

            resultado_dobra = metricas(y_real, predicoes)

            relatorios.append({
                "dobra": numero,
                **resultado_dobra,
            })

            print(
                f"  Dobra {numero}: "
                f"RMSE={resultado_dobra['rmse']:.2f}",
                flush=True,
            )

        resultados[algoritmo] = {
            **metricas(reais, previstas),
            "dobras": relatorios,
        }

        print(
            f"  RMSE agregado: "
            f"{resultados[algoritmo]['rmse']:.2f}",
            flush=True,
        )

    vencedor = min(
        resultados,
        key=lambda nome: resultados[nome]["rmse"],
    )

    return vencedor, resultados


def avaliar_externamente(modelo, esquema, teste):
    x_teste = transformar_atributos(teste, esquema)

    predicoes = np.maximum(
        0,
        np.asarray(
            modelo.predict(x_teste),
            dtype=float,
        ),
    )

    resultado = teste[
        [
            "loja_id",
            "data_origem",
            "data",
            "horizonte_dias",
            "vendas",
        ]
    ].copy()

    resultado["vendas_previstas"] = predicoes

    por_horizonte = {}

    for horizonte in range(1, 8):
        grupo = resultado.loc[
            resultado["horizonte_dias"] == horizonte
        ]

        por_horizonte[str(horizonte)] = metricas(
            grupo["vendas"],
            grupo["vendas_previstas"],
        )

    agregado = metricas(
        resultado["vendas"],
        resultado["vendas_previstas"],
    )

    return resultado, agregado, por_horizonte


def gerar_previsoes_semanais(dados, modelo, esquema):
    resultados = []

    for loja_id in dados["loja_id"].dropna().unique():
        futuras = gerar_entradas_futuras_loja(
            dados,
            int(loja_id),
        )

        x_futuro = transformar_atributos(
            futuras,
            esquema,
        )

        predicoes = np.maximum(
            0,
            np.asarray(
                modelo.predict(x_futuro),
                dtype=float,
            ),
        )

        previsoes = []

        for indice, (_, linha) in enumerate(
            futuras.iterrows()
        ):
            previsoes.append({
                "horizonte_dias": int(
                    linha["horizonte_dias"]
                ),
                "data": linha["data"].strftime(
                    "%Y-%m-%d"
                ),
                "vendas_previstas": round(
                    float(predicoes[indice]),
                    2,
                ),
            })

        resultados.append({
            "loja_id": int(loja_id),
            "data_origem": futuras[
                "data_origem"
            ].iloc[0].strftime("%Y-%m-%d"),
            "previsoes": previsoes,
            "total_semanal_previsto": round(
                sum(
                    item["vendas_previstas"]
                    for item in previsoes
                ),
                2,
            ),
        })

    return resultados


def executar_pipeline_multihorizonte(execucao_id):
    inicio = time.perf_counter()

    pasta = pasta_da_execucao(execucao_id)

    destino = pasta / "multihorizonte"

    if destino.exists():
        raise FileExistsError(
            "Esta execução já possui resultados "
            "multihorizonte. Não serão sobrescritos."
        )

    print("\nCarregando histórico...", flush=True)

    # Inicialmente usamos o dataset original,
    # para manter comparação reproduzível.
    dados = pd.read_csv(
        pasta / "dataset.csv",
        dtype={"loja_id": "string"},
    )

    dados["data"] = pd.to_datetime(
        dados["data"],
        format="%Y-%m-%d",
    )

    dados["vendas"] = pd.to_numeric(
        dados["vendas"],
        errors="raise",
    )

    if dados["vendas"].isna().any():
        raise ValueError("Existem vendas inválidas.")

    # Limite provisório para proteger a memória
    # durante a validação inicial do pipeline.
    if len(dados) > 5000:
        raise ValueError(
            "Este piloto aceita no máximo 5.000 linhas. "
            "O processamento em lotes virá depois."
        )

    print(
        f"Dataset: {len(dados)} linhas, "
        f"{dados['loja_id'].nunique()} lojas",
        flush=True,
    )

    exemplos = preparar_exemplos(dados)

    treino, teste, relatorio_divisao = (
        dividir_dataset_multihorizonte(exemplos)
    )

    divisoes, relatorio_dobras = (
        criar_dobras_multihorizonte(treino)
    )

    print(
        f"Treino: {len(treino)} | "
        f"Teste externo: {len(teste)} | "
        f"Dobras: {len(divisoes)}",
        flush=True,
    )

    vencedor, selecao = competir_modelos(
        treino,
        divisoes,
    )

    print(
        f"\nVencedor: {vencedor}",
        flush=True,
    )

    # Preprocessamento final ajustado SOMENTE
    # sobre os dados de treinamento externo.
    esquema_final = ajustar_preprocessamento(treino)

    x_treino = transformar_atributos(
        treino,
        esquema_final,
    )

    modelo_final = criar_modelo(
        vencedor,
        CONFIGURACOES[vencedor],
    )

    modelo_final.fit(
        x_treino,
        treino["vendas"].to_numpy(dtype=float),
    )

    previsoes_teste, agregado, por_horizonte = (
        avaliar_externamente(
            modelo_final,
            esquema_final,
            teste,
        )
    )

    previsoes_futuras = gerar_previsoes_semanais(
        dados,
        modelo_final,
        esquema_final,
    )

    relatorio = {
        "tipo": "piloto_multihorizonte",
        "horizontes": list(range(1, 8)),
        "otimizacao_flaml": False,
        "algoritmo_vencedor": vencedor,
        "configuracao": CONFIGURACOES[vencedor],
        "selecao_temporal": selecao,
        "divisao_externa": relatorio_divisao,
        "validacao_interna": relatorio_dobras,
        "avaliacao_externa": {
            "agregado": agregado,
            "por_horizonte": por_horizonte,
        },
        "tempo_total_segundos": round(
            time.perf_counter() - inicio,
            2,
        ),
    }

    destino.mkdir(parents=True, exist_ok=False)

    joblib.dump(
        {
            "algoritmo": vencedor,
            "modelo": modelo_final,
            "preprocessamento": esquema_final,
            "horizontes": list(range(1, 8)),
            "tipo": "multihorizonte",
        },
        destino / "modelo_multihorizonte.joblib",
    )

    previsoes_teste.to_csv(
        destino / "previsoes_teste.csv",
        index=False,
    )

    (destino / "relatorio.json").write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (destino / "previsoes_futuras.json").write_text(
        json.dumps(
            previsoes_futuras,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nAVALIAÇÃO EXTERNA", flush=True)
    print(
        f"RMSE agregado: {agregado['rmse']:.2f}",
        flush=True,
    )
    print(
        f"MAE agregado: {agregado['mae']:.2f}",
        flush=True,
    )

    for horizonte, valores in por_horizonte.items():
        print(
            f"Dia {horizonte}: "
            f"RMSE={valores['rmse']:.2f} | "
            f"MAE={valores['mae']:.2f}",
            flush=True,
        )

    print("\nPREVISÕES FUTURAS", flush=True)

    for loja in previsoes_futuras:
        print(
            f"\nLoja {loja['loja_id']}",
            flush=True,
        )

        for previsao in loja["previsoes"]:
            print(
                f"  {previsao['data']}: "
                f"{previsao['vendas_previstas']:.2f}",
                flush=True,
            )

        print(
            "  TOTAL SEMANAL: "
            f"{loja['total_semanal_previsto']:.2f}",
            flush=True,
        )

    print(
        f"\nArquivos salvos em: {destino}",
        flush=True,
    )

    return relatorio
