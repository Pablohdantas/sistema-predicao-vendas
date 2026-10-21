
"""
Pipeline AutoML multihorizonte: previsão dos dias 1 a 7.

- Engenharia de atributos sem vendas futuras.
- Divisão temporal externa.
- Validação temporal expansiva.
- Otimização com FLAML Tune.
- Competição LightGBM x XGBoost.
- Seleção pelo menor RMSE de validação.
- Avaliação externa por horizonte.
- Previsão semanal por loja.
- Persistência independente do modelo convencional.

O teste externo não participa da otimização.
"""

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flaml import tune

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
from backend.app.services.treinamento_service import (
    pasta_da_execucao,
)


SEMENTE = 42
HORIZONTES = list(range(1, 8))

# Valores conservadores para o primeiro uso no notebook.
TENTATIVAS_POR_ALGORITMO = 3
TEMPO_POR_ALGORITMO = 30

# Limite de segurança desta etapa.
# Para bases grandes, será necessário processamento
# em lotes, sem expandir todas as linhas em memória.
LIMITE_LINHAS_PILOTO = 5000


CONFIGURACOES_INICIAIS = {
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
    real = np.asarray(
        y_real,
        dtype=float,
    ).reshape(-1)

    previsto = np.asarray(
        y_previsto,
        dtype=float,
    ).reshape(-1)

    if (
        real.size == 0
        or real.size != previsto.size
        or not np.isfinite(real).all()
        or not np.isfinite(previsto).all()
    ):
        raise ValueError(
            "Valores inválidos no cálculo das métricas."
        )

    erros = real - previsto

    return {
        "rmse": float(
            np.sqrt(np.mean(np.square(erros)))
        ),
        "mae": float(
            np.mean(np.abs(erros))
        ),
        "quantidade": int(len(real)),
    }


def normalizar_configuracao(algoritmo, configuracao):
    """
    Normaliza valores da busca para tipos aceitos
    pelos modelos e pelo formato JSON.
    """

    resultado = {
        "n_estimators": int(
            configuracao["n_estimators"]
        ),
        "learning_rate": float(
            configuracao["learning_rate"]
        ),
        "max_depth": int(
            configuracao["max_depth"]
        ),
        "subsample": float(
            configuracao["subsample"]
        ),
        "colsample_bytree": float(
            configuracao["colsample_bytree"]
        ),
    }

    if algoritmo == "lgbm":
        resultado["num_leaves"] = int(
            configuracao["num_leaves"]
        )

    return resultado


def espaco_busca(algoritmo):
    """
    Espaço de hiperparâmetros do FLAML.

    Num_samples e time_budget_s limitam a busca
    para reduzir o consumo de recursos.
    """

    espaco = {
        "n_estimators": tune.randint(60, 241),
        "learning_rate": tune.loguniform(
            0.015,
            0.20,
        ),
        "max_depth": tune.randint(3, 9),
        "subsample": tune.uniform(0.75, 1.0),
        "colsample_bytree": tune.uniform(
            0.75,
            1.0,
        ),
    }

    if algoritmo == "lgbm":
        espaco["num_leaves"] = tune.randint(8, 64)

    return espaco


def preparar_exemplos(dados):
    """
    Constrói exemplos dos sete horizontes para
    todas as lojas do dataset.
    """

    partes = []

    for identificador in dados["loja_id"].dropna().unique():
        parte = gerar_exemplos_treinamento_loja(
            dados,
            int(identificador),
        )

        partes.append(parte)

    if not partes:
        raise ValueError(
            "Nenhuma loja válida encontrada."
        )

    resultado = pd.concat(
        partes,
        ignore_index=True,
    )

    del partes
    gc.collect()

    if resultado.empty:
        raise ValueError(
            "Nenhum exemplo de treinamento disponível."
        )

    return resultado


def avaliar_configuracao(
    algoritmo,
    configuracao,
    treino_externo,
    divisoes,
):
    """
    Avalia a mesma configuração em todas as dobras.

    Para cada dobra:
    - Ajusta o pré-processamento somente no treino.
    - Treina o modelo.
    - Prevê sobre a validação.
    - Acumula erros para calcular RMSE agregado.
    """

    configuracao = normalizar_configuracao(
        algoritmo,
        configuracao,
    )

    soma_quadrados = 0.0
    soma_absolutos = 0.0
    total = 0

    relatorios = []

    for numero, (idx_treino, idx_validacao) in enumerate(
        divisoes,
        start=1,
    ):
        treino = treino_externo.iloc[idx_treino]
        validacao = treino_externo.iloc[idx_validacao]

        if (
            treino["data"].max()
            > validacao["data_origem"].min()
        ):
            raise ValueError(
                f"Vazamento temporal na dobra {numero}."
            )

        esquema = ajustar_preprocessamento(treino)

        x_treino = transformar_atributos(
            treino,
            esquema,
        )

        x_validacao = transformar_atributos(
            validacao,
            esquema,
        )

        y_treino = treino["vendas"].to_numpy(
            dtype=float
        )

        y_validacao = validacao["vendas"].to_numpy(
            dtype=float
        )

        modelo = criar_modelo(
            algoritmo,
            configuracao,
        )

        modelo.fit(
            x_treino,
            y_treino,
        )

        previsoes = np.asarray(
            modelo.predict(x_validacao),
            dtype=float,
        )

        if not np.isfinite(previsoes).all():
            raise ValueError(
                "O modelo retornou previsões inválidas."
            )

        previsoes = np.maximum(
            previsoes,
            0,
        )

        erros = y_validacao - previsoes

        soma_quadrados += float(
            np.sum(np.square(erros))
        )

        soma_absolutos += float(
            np.sum(np.abs(erros))
        )

        total += len(erros)

        relatorios.append({
            "dobra": numero,
            **metricas(y_validacao, previsoes),
        })

        del (
            modelo,
            x_treino,
            x_validacao,
            y_treino,
            y_validacao,
            previsoes,
            erros,
        )

        gc.collect()

    if total <= 0:
        raise ValueError(
            "Nenhuma previsão gerada nas dobras."
        )

    return {
        "rmse": float(
            np.sqrt(soma_quadrados / total)
        ),
        "mae": float(
            soma_absolutos / total
        ),
        "quantidade": total,
        "dobras": relatorios,
    }


def otimizar_algoritmo(
    algoritmo,
    treino_externo,
    divisoes,
    tentativas,
    tempo_por_algoritmo,
):
    """
    Executa busca FLAML e mantém a melhor configuração
    avaliada durante a competição.
    """

    print(
        f"\nOtimizando {algoritmo} com FLAML...",
        flush=True,
    )

    melhor_configuracao = normalizar_configuracao(
        algoritmo,
        CONFIGURACOES_INICIAIS[algoritmo],
    )

    melhor_resultado = avaliar_configuracao(
        algoritmo,
        melhor_configuracao,
        treino_externo,
        divisoes,
    )

    avaliacoes = [{
        "origem": "configuracao_inicial",
        "configuracao": melhor_configuracao,
        "rmse": melhor_resultado["rmse"],
        "mae": melhor_resultado["mae"],
    }]

    print(
        "  Configuração inicial: "
        f"RMSE={melhor_resultado['rmse']:.2f}",
        flush=True,
    )

    def objetivo(configuracao):
        nonlocal melhor_configuracao
        nonlocal melhor_resultado

        candidata = normalizar_configuracao(
            algoritmo,
            configuracao,
        )

        resultado = avaliar_configuracao(
            algoritmo,
            candidata,
            treino_externo,
            divisoes,
        )

        avaliacoes.append({
            "origem": "flaml",
            "configuracao": candidata,
            "rmse": resultado["rmse"],
            "mae": resultado["mae"],
        })

        if resultado["rmse"] < melhor_resultado["rmse"]:
            melhor_resultado = resultado
            melhor_configuracao = candidata

        return {
            "rmse": resultado["rmse"],
        }

    # Cada algoritmo recebe o mesmo número máximo
    # de tentativas e orçamento de tempo.
    tune.run(
        objetivo,
        config=espaco_busca(algoritmo),
        metric="rmse",
        mode="min",
        num_samples=tentativas,
        time_budget_s=tempo_por_algoritmo,
        low_cost_partial_config=(
            CONFIGURACOES_INICIAIS[algoritmo]
        ),
        use_ray=False,
        verbose=0,
    )

    print(
        f"  Melhor RMSE: {melhor_resultado['rmse']:.2f}",
        flush=True,
    )

    return {
        "algoritmo": algoritmo,
        "configuracao": melhor_configuracao,
        "resultado": melhor_resultado,
        "tentativas_avaliadas": len(avaliacoes),
        "avaliacoes": avaliacoes,
    }


def competir_modelos(
    treino_externo,
    divisoes,
    tentativas,
    tempo_por_algoritmo,
):
    """
    Otimiza ambos os algoritmos e seleciona o
    menor RMSE de validação temporal.
    """

    resultados = {}

    for algoritmo in ("lgbm", "xgboost"):
        resultados[algoritmo] = otimizar_algoritmo(
            algoritmo,
            treino_externo,
            divisoes,
            tentativas,
            tempo_por_algoritmo,
        )

    vencedor = min(
        resultados,
        key=lambda nome: (
            resultados[nome]["resultado"]["rmse"]
        ),
    )

    return vencedor, resultados


def avaliar_externamente(
    modelo,
    esquema,
    teste,
):
    """
    Avalia o vencedor no teste externo reservado.

    Calcula métricas agregadas e separadas
    para os horizontes de 1 a 7.
    """

    x_teste = transformar_atributos(
        teste,
        esquema,
    )

    previsoes = np.asarray(
        modelo.predict(x_teste),
        dtype=float,
    )

    if not np.isfinite(previsoes).all():
        raise ValueError(
            "Previsões externas inválidas."
        )

    previsoes = np.maximum(previsoes, 0)

    resultado = teste[
        [
            "loja_id",
            "data_origem",
            "data",
            "horizonte_dias",
            "vendas",
        ]
    ].copy()

    resultado["vendas_previstas"] = previsoes

    agregado = metricas(
        resultado["vendas"],
        resultado["vendas_previstas"],
    )

    por_horizonte = {}

    for horizonte in HORIZONTES:
        grupo = resultado.loc[
            resultado["horizonte_dias"] == horizonte
        ]

        por_horizonte[str(horizonte)] = metricas(
            grupo["vendas"],
            grupo["vendas_previstas"],
        )

    return (
        resultado,
        agregado,
        por_horizonte,
    )


def gerar_previsoes_semanais(
    dados,
    modelo,
    esquema,
):
    """
    Produz sete estimativas por loja a partir
    do último registro histórico observado.
    """

    resultados = []

    for identificador in dados["loja_id"].dropna().unique():
        loja_id = int(identificador)

        futuras = gerar_entradas_futuras_loja(
            dados,
            loja_id,
        )

        x_futuro = transformar_atributos(
            futuras,
            esquema,
        )

        valores = np.asarray(
            modelo.predict(x_futuro),
            dtype=float,
        )

        if (
            len(valores) != 7
            or not np.isfinite(valores).all()
        ):
            raise ValueError(
                f"Previsões inválidas para loja {loja_id}."
            )

        valores = np.maximum(valores, 0)

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
                    float(valores[indice]),
                    2,
                ),
            })

        resultados.append({
            "loja_id": loja_id,
            "data_origem": (
                futuras["data_origem"]
                .iloc[0]
                .strftime("%Y-%m-%d")
            ),
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


def executar_pipeline_multihorizonte(
    execucao_id,
    tentativas=TENTATIVAS_POR_ALGORITMO,
    tempo_por_algoritmo=TEMPO_POR_ALGORITMO,
):
    """
    Executa o AutoML semanal completo.

    O modelo convencional da execução permanece
    preservado. O teste externo só é avaliado
    depois da escolha dos hiperparâmetros.
    """

    inicio = time.perf_counter()

    if not 1 <= tentativas <= 30:
        raise ValueError(
            "Tentativas devem estar entre 1 e 30."
        )

    if not 1 <= tempo_por_algoritmo <= 1800:
        raise ValueError(
            "Tempo deve estar entre 1 e 1800 segundos."
        )

    pasta = pasta_da_execucao(execucao_id)

    destino = pasta / "multihorizonte"

    if destino.exists():
        raise FileExistsError(
            "O modelo multihorizonte já existe. "
            "Os arquivos não serão sobrescritos."
        )

    print(
        "\n=== AUTOML MULTIHORIZONTE ===",
        flush=True,
    )

    dados = pd.read_csv(
        pasta / "dataset.csv",
        dtype={"loja_id": "string"},
    )

    if len(dados) > LIMITE_LINHAS_PILOTO:
        raise ValueError(
            "O treinamento multihorizonte aceita "
            f"até {LIMITE_LINHAS_PILOTO} registros "
            "nesta etapa. O suporte a bases maiores "
            "exige processamento em lotes."
        )

    dados["data"] = pd.to_datetime(
        dados["data"],
        format="%Y-%m-%d",
        errors="raise",
    )

    dados["vendas"] = pd.to_numeric(
        dados["vendas"],
        errors="raise",
    )

    if (
        dados["vendas"].isna().any()
        or not np.isfinite(dados["vendas"]).all()
    ):
        raise ValueError(
            "O dataset possui vendas inválidas."
        )

    print(
        f"Registros: {len(dados)} | "
        f"Lojas: {dados['loja_id'].nunique()}",
        flush=True,
    )

    exemplos = preparar_exemplos(dados)

    treino, teste, relatorio_divisao = (
        dividir_dataset_multihorizonte(
            exemplos
        )
    )

    del exemplos
    gc.collect()

    divisoes, relatorio_dobras = (
        criar_dobras_multihorizonte(
            treino
        )
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
        tentativas,
        tempo_por_algoritmo,
    )

    configuracao_vencedora = selecao[
        vencedor
    ]["configuracao"]

    print(
        f"\nVencedor: {vencedor.upper()}",
        flush=True,
    )

    # Treina o vencedor somente nos exemplos
    # anteriores ao período de teste externo.
    esquema_final = ajustar_preprocessamento(
        treino
    )

    x_treino = transformar_atributos(
        treino,
        esquema_final,
    )

    modelo_final = criar_modelo(
        vencedor,
        configuracao_vencedora,
    )

    modelo_final.fit(
        x_treino,
        treino["vendas"].to_numpy(dtype=float),
    )

    del x_treino
    gc.collect()

    previsoes_teste, agregado, por_horizonte = (
        avaliar_externamente(
            modelo_final,
            esquema_final,
            teste,
        )
    )

    # Gera previsões futuras usando o modelo salvo
    # com o histórico original desta execução.
    # Na API, previsões posteriores usam o histórico
    # atualizado, sem retreinamento.
    previsoes_futuras = gerar_previsoes_semanais(
        dados,
        modelo_final,
        esquema_final,
    )

    relatorio = {
        "tipo": "automl_multihorizonte",
        "horizontes": HORIZONTES,
        "otimizacao_flaml": True,
        "algoritmo_vencedor": vencedor,
        "configuracao": configuracao_vencedora,
        "tentativas_por_algoritmo": tentativas,
        "tempo_por_algoritmo": tempo_por_algoritmo,
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

    # Mantém o mesmo tipo de artefato que o
    # endpoint de previsão semanal já conhece.
    artefato = {
        "algoritmo": vencedor,
        "modelo": modelo_final,
        "preprocessamento": esquema_final,
        "horizontes": HORIZONTES,
        "tipo": "multihorizonte",
    }

    destino.mkdir(
        parents=True,
        exist_ok=False,
    )

    joblib.dump(
        artefato,
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

    print(
        "\n=== AVALIAÇÃO EXTERNA ===",
        flush=True,
    )

    print(
        f"RMSE: {agregado['rmse']:.2f} | "
        f"MAE: {agregado['mae']:.2f}",
        flush=True,
    )

    for horizonte, valores in por_horizonte.items():
        print(
            f"Dia {horizonte}: "
            f"RMSE={valores['rmse']:.2f} | "
            f"MAE={valores['mae']:.2f}",
            flush=True,
        )

    print(
        "\n=== PREVISÕES SEMANAIS ===",
        flush=True,
    )

    for loja in previsoes_futuras:
        print(
            f"Loja {loja['loja_id']}: "
            f"total={loja['total_semanal_previsto']:.2f}",
            flush=True,
        )

    print(
        f"\nArquivos salvos em: {destino}",
        flush=True,
    )

    return relatorio
