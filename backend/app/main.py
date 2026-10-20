
"""
Aplicação FastAPI do Sistema de Predição de Vendas.
"""

from fastapi import FastAPI

from backend.app.routers.previsoes import (
    router as previsoes_router
)
from backend.app.routers.previsoes_automaticas import (
    router as previsoes_automaticas_router
)
from backend.app.routers.datasets import (
    router as datasets_router
)
from backend.app.routers.treinamentos import (
    router as treinamentos_router
)
from backend.app.routers.avaliacoes import (
    router as avaliacoes_router
)
from backend.app.routers.historico import (
    router as historico_router
)
from backend.app.routers.execucoes import (
    router as execucoes_router
)

app = FastAPI(
    title="Sistema de Predição de Vendas",
    description=(
        "API para upload de datasets, treinamento AutoML "
        "com LightGBM e XGBoost, avaliação temporal, "
        "atualização do histórico de vendas, "
        "previsões e explicabilidade SHAP."
    ),
    version="1.4.0"
)


@app.get("/")
def inicio():
    return {
        "mensagem": "API de previsão de vendas funcionando!",
        "status": "online"
    }


@app.get("/health")
def verificar_saude():
    return {"status": "ok"}


app.include_router(datasets_router)
app.include_router(treinamentos_router)
app.include_router(avaliacoes_router)
app.include_router(historico_router)
app.include_router(previsoes_automaticas_router)
app.include_router(previsoes_router)
app.include_router(execucoes_router)