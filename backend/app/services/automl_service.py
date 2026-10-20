
"""
Motor AutoML dinâmico para previsão de vendas.

Fluxo:
1. Carrega os atributos da execução.
2. Reserva o teste externo.
3. Cria quatro dobras temporais expansivas.
4. Otimiza LightGBM e XGBoost com FLAML Tune.
5. Avalia cada configuração nas mesmas quatro dobras.
6. Seleciona o menor RMSE agregado.
7. Em modo definitivo, treina o vencedor com todo
   o conjunto de treinamento externo.
8. Salva modelo, preprocessamento e relatório.

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
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from backend.app.services.validacao_temporal_adaptativa_service import criar_divisoes

from backend.app.services.divisao_temporal_service import (
    dividir_dataset_temporalmente
)

from backend.app.services.preprocessamento_service import (
    ajustar_preprocessamento,
    transformar_atributos
)


ALGORITMOS = ("lgbm", "xgboost")
SEMENTE = 42
HORIZONTE = 7
N_THREADS = 2


def converter_json(valor):
    """Converte tipos NumPy para tipos compatíveis com JSON."""

    if isinstance(valor, np.generic):
        return valor.item()

    if isinstance(valor, np.ndarray):
        return valor.tolist()

    if isinstance(valor, (pd.Timestamp, Path)):
        return str(valor)

    raise TypeError(
        f"Tipo não serializável: {type(valor).__name__}"
    )


def criar_modelo(algoritmo: str, configuracao: dict):
    """Instancia o algoritmo com a configuração recebida."""

    parametros = {
        "n_estimators": int(configuracao["n_estimators"]),
        "learning_rate": float(configuracao["learning_rate"]),
        "max_depth": int(configuracao["max_depth"]),
        "subsample": float(configuracao["subsample"]),
        "colsample_bytree": float(
            configuracao["colsample_bytree"]
        ),
        "n_jobs": N_THREADS,
        "random_state": SEMENTE,
    }

    if algoritmo == "lgbm":
        return LGBMRegressor(
            **parametros,
            num_leaves=int(configuracao["num_leaves"]),
            verbosity=-1
        )

    if algoritmo == "xgboost":
        return XGBRegressor(
            **parametros,
            objective="reg:squarederror",
            tree_method="hist",
            verbosity=0
        )

    raise ValueError(
        f"Algoritmo desconhecido: {algoritmo}"
    )


def carregar_execucao(pasta_execucao: Path):
    """
    Carrega o dataset e cria as dobras internas
    usando exclusivamente o treinamento externo.
    """

    pasta_execucao = Path(pasta_execucao)
    caminho = pasta_execucao / "atributos_h7.csv"

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    dados = pd.read_csv(
        caminho,
        parse_dates=["data", "data_origem"]
    )

    treino_externo, teste_externo, relatorio_divisao = (
        dividir_dataset_temporalmente(dados)
    )

    treino_externo = treino_externo.reset_index(drop=True)

    divisoes, relatorio_dobras = criar_divisoes(
        treino_externo[
            ["loja_id", "data", "data_origem"]
        ]
    )

    del dados
    gc.collect()

    return (
        treino_externo,
        len(teste_externo),
        divisoes,
        relatorio_divisao,
        relatorio_dobras
    )


def preparar_dobra(
    treino_externo: pd.DataFrame,
    indices: tuple,
    modo_teste: bool
):
    """
    Prepara uma dobra, ajustando o preprocessamento
    somente com seus dados de treinamento.
    """

    idx_treino, idx_validacao = indices

    treino = treino_externo.iloc[idx_treino].copy()
    validacao = treino_externo.iloc[idx_validacao].copy()

    if modo_teste:
        if len(treino) > 2000:
            treino = treino.sample(
                n=2000,
                random_state=SEMENTE
            )

        if len(validacao) > 1000:
            validacao = validacao.sample(
                n=1000,
                random_state=SEMENTE
            )

    treino = treino.sort_values(
        ["data", "loja_id"]
    ).reset_index(drop=True)

    validacao = validacao.sort_values(
        ["data", "loja_id"]
    ).reset_index(drop=True)

    if treino.empty or validacao.empty:
        raise ValueError("Dobra sem observações.")

    if (
        treino["data"].max()
        > validacao["data_origem"].min()
    ):
        raise ValueError(
            "Vazamento temporal detectado."
        )

    preprocessamento = ajustar_preprocessamento(treino)

    x_treino = transformar_atributos(
        treino,
        preprocessamento
    )

    x_validacao = transformar_atributos(
        validacao,
        preprocessamento
    )

    if not x_treino.columns.equals(x_validacao.columns):
        raise ValueError(
            "Colunas inconsistentes entre treino e validação."
        )

    y_treino = treino["vendas"].to_numpy(dtype=float)
    y_validacao = validacao["vendas"].to_numpy(dtype=float)

    if (
        not np.isfinite(y_treino).all()
        or not np.isfinite(y_validacao).all()
    ):
        raise ValueError("Valores-alvo inválidos.")

    return (
        x_treino,
        y_treino,
        x_validacao,
        y_validacao
    )


def avaliar_configuracao(
    algoritmo: str,
    configuracao: dict,
    treino_externo: pd.DataFrame,
    divisoes: list,
    modo_teste: bool
) -> dict:
    """
    Avalia uma mesma configuração em todas as dobras.

    Processa uma dobra de cada vez para reduzir
    o consumo de memória.
    """

    soma_quadrados = 0.0
    soma_absolutos = 0.0
    total_observacoes = 0
    relatorios = []

    for numero, indices in enumerate(divisoes, start=1):
        (
            x_treino,
            y_treino,
            x_validacao,
            y_validacao
        ) = preparar_dobra(
            treino_externo,
            indices,
            modo_teste
        )

        modelo = criar_modelo(
            algoritmo,
            configuracao
        )

        modelo.fit(x_treino, y_treino)

        previsoes = np.asarray(
            modelo.predict(x_validacao),
            dtype=float
        )

        erros = y_validacao - previsoes

        if not np.isfinite(erros).all():
            raise ValueError(
                f"Previsões inválidas na dobra {numero}."
            )

        quantidade = len(erros)

        quadrados = float(
            np.sum(np.square(erros))
        )

        absolutos = float(
            np.sum(np.abs(erros))
        )

        soma_quadrados += quadrados
        soma_absolutos += absolutos
        total_observacoes += quantidade

        relatorios.append({
            "dobra": numero,
            "rmse": float(
                np.sqrt(quadrados / quantidade)
            ),
            "mae": float(
                absolutos / quantidade
            ),
            "observacoes": quantidade
        })

        del (
            modelo,
            previsoes,
            erros,
            x_treino,
            x_validacao,
            y_treino,
            y_validacao
        )
        gc.collect()

    return {
        "rmse_agregado": float(
            np.sqrt(soma_quadrados / total_observacoes)
        ),
        "mae_agregado": float(
            soma_absolutos / total_observacoes
        ),
        "quantidade_previsoes": total_observacoes,
        "dobras": relatorios
    }


def otimizar_algoritmo(
    algoritmo: str,
    treino_externo: pd.DataFrame,
    divisoes: list,
    modo_teste: bool,
    tentativas: int,
    tempo_limite: int
) -> dict:
    """Busca a melhor configuração usando FLAML Tune."""

    espaco = {
        "n_estimators": tune.randint(50, 251),
        "learning_rate": tune.loguniform(0.01, 0.2),
        "max_depth": tune.randint(3, 9),
        "subsample": tune.uniform(0.7, 1.0),
        "colsample_bytree": tune.uniform(0.7, 1.0),
    }

    inicial = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    if algoritmo == "lgbm":
        espaco["num_leaves"] = tune.randint(8, 33)
        inicial["num_leaves"] = 15

    historico = []

    def objetivo(configuracao):
        inicio = time.perf_counter()

        avaliacao = avaliar_configuracao(
            algoritmo=algoritmo,
            configuracao=configuracao,
            treino_externo=treino_externo,
            divisoes=divisoes,
            modo_teste=modo_teste
        )

        registro = {
            "tentativa": len(historico) + 1,
            "configuracao": dict(configuracao),
            "rmse_agregado": avaliacao["rmse_agregado"],
            "mae_agregado": avaliacao["mae_agregado"],
            "dobras": avaliacao["dobras"],
            "tempo_segundos": round(
                time.perf_counter() - inicio,
                2
            )
        }

        historico.append(registro)

        print(
            f"  Tentativa {registro['tentativa']}: "
            f"RMSE={registro['rmse_agregado']:.2f}",
            flush=True
        )

        return {
            "rmse": avaliacao["rmse_agregado"]
        }

    inicio = time.perf_counter()

    analise = tune.run(
        objetivo,
        config=espaco,
        points_to_evaluate=[inicial],
        metric="rmse",
        mode="min",
        num_samples=tentativas,
        time_budget_s=tempo_limite,
        use_ray=False,
        verbose=0
    )

    if analise.best_config is None:
        raise RuntimeError(
            f"Nenhuma configuração válida para {algoritmo}."
        )

    melhor_configuracao = dict(analise.best_config)

    # Aproveita a avaliação já realizada pelo FLAML.
    # Evita executar novamente as quatro dobras.
    melhor_registro = min(
        historico,
        key=lambda item: item["rmse_agregado"]
    )

    return {
        "algoritmo": algoritmo,
        "melhor_configuracao": melhor_configuracao,
        "rmse_agregado": melhor_registro["rmse_agregado"],
        "mae_agregado": melhor_registro["mae_agregado"],
        "dobras": melhor_registro["dobras"],
        "tentativas_realizadas": len(historico),
        "historico": historico,
        "tempo_segundos": round(
            time.perf_counter() - inicio,
            2
        )
    }


def treinar_modelo_final(
    algoritmo: str,
    configuracao: dict,
    treino_externo: pd.DataFrame,
    pasta_execucao: Path
) -> Path:
    """
    Ajusta o preprocessamento no treinamento externo
    completo e treina o modelo vencedor.
    """

    print(
        f"\nTreinando modelo final: {algoritmo.upper()}",
        flush=True
    )

    preprocessamento = ajustar_preprocessamento(
        treino_externo
    )

    x_treino = transformar_atributos(
        treino_externo,
        preprocessamento
    )

    y_treino = treino_externo["vendas"].to_numpy(
        dtype=float
    )

    modelo = criar_modelo(
        algoritmo,
        configuracao
    )

    modelo.fit(x_treino, y_treino)

    artefato = {
        "algoritmo": algoritmo,
        "modelo": modelo,
        "preprocessamento": preprocessamento,
        "configuracao": configuracao,
        "colunas_modelo": x_treino.columns.tolist(),
        "horizonte_dias": HORIZONTE,
        "linhas_treinamento": len(x_treino)
    }

    caminho = pasta_execucao / "modelo_vencedor.joblib"

    joblib.dump(artefato, caminho)

    del modelo, x_treino, y_treino
    gc.collect()

    return caminho


def executar_pipeline_automl(
    pasta_execucao: Path,
    modo_teste: bool = True,
    tentativas: int | None = None,
    tempo_por_algoritmo: int | None = None
) -> dict:
    """
    Executa o pipeline completo de seleção AutoML.

    Em modo de teste:
        usa amostras reduzidas;
        não treina nem salva modelo definitivo.

    Em modo definitivo:
        usa todo o treinamento externo;
        seleciona e salva o vencedor.
    """

    pasta_execucao = Path(pasta_execucao)

    if not pasta_execucao.is_dir():
        raise FileNotFoundError(
            f"Execução não encontrada: {pasta_execucao}"
        )

    if tentativas is None:
        tentativas = 2 if modo_teste else 15

    if tempo_por_algoritmo is None:
        tempo_por_algoritmo = 60 if modo_teste else 360

    if tentativas < 1 or tempo_por_algoritmo < 1:
        raise ValueError(
            "Tentativas e tempo devem ser positivos."
        )

    inicio = time.perf_counter()

    (
        treino_externo,
        quantidade_teste,
        divisoes,
        relatorio_divisao,
        relatorio_dobras
    ) = carregar_execucao(pasta_execucao)

    resultados = []

    for algoritmo in ALGORITMOS:
        print(
            f"\n=== OTIMIZANDO {algoritmo.upper()} ===",
            flush=True
        )

        resultado = otimizar_algoritmo(
            algoritmo=algoritmo,
            treino_externo=treino_externo,
            divisoes=divisoes,
            modo_teste=modo_teste,
            tentativas=tentativas,
            tempo_limite=tempo_por_algoritmo
        )

        resultados.append(resultado)

    melhor = min(
        resultados,
        key=lambda item: item["rmse_agregado"]
    )

    algoritmo_vencedor = (
        None if modo_teste else melhor["algoritmo"]
    )

    caminho_modelo = None

    if not modo_teste:
        caminho_modelo = treinar_modelo_final(
            algoritmo=melhor["algoritmo"],
            configuracao=melhor["melhor_configuracao"],
            treino_externo=treino_externo,
            pasta_execucao=pasta_execucao
        )

    relatorio = {
        "execucao_id": pasta_execucao.name,
        "modo": (
            "teste_integracao"
            if modo_teste
            else "treinamento_definitivo"
        ),
        "metodo": "flaml_tune_validacao_temporal",
        "horizonte_dias": HORIZONTE,
        "semente": SEMENTE,
        "criterio_selecao": "rmse_agregado",
        "data_corte": relatorio_divisao["data_corte"],
        "linhas_treino_externo": len(treino_externo),
        "linhas_teste_externo": quantidade_teste,
        "quantidade_dobras": len(divisoes),
        "divisoes_temporais": relatorio_dobras,
        "tentativas_por_algoritmo": tentativas,
        "tempo_limite_por_algoritmo": tempo_por_algoritmo,
        "resultados": resultados,
        "melhor_algoritmo_na_validacao": melhor["algoritmo"],
        "algoritmo_vencedor": algoritmo_vencedor,
        "configuracao_selecionada": (
            melhor["melhor_configuracao"]
            if not modo_teste
            else None
        ),
        "rmse_selecao": melhor["rmse_agregado"],
        "mae_selecao": melhor["mae_agregado"],
        "modelo_salvo": (
            caminho_modelo.name
            if caminho_modelo is not None
            else None
        ),
        "tempo_total_segundos": round(
            time.perf_counter() - inicio,
            2
        )
    }

    nome_relatorio = (
        "competicao_automl_teste.json"
        if modo_teste
        else "competicao_automl.json"
    )

    caminho_relatorio = pasta_execucao / nome_relatorio

    caminho_relatorio.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2,
            default=converter_json
        ),
        encoding="utf-8"
    )

    print(
        f"\nRelatório salvo: {caminho_relatorio}",
        flush=True
    )

    if not modo_teste:
        print(
            f"Modelo salvo: {caminho_modelo}",
            flush=True
        )

    return relatorio
