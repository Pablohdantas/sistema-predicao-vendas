
from fastapi import FastAPI

from backend.app.routers.previsoes import router as previsoes_router
from backend.app.routers.datasets import router as datasets_router

app = FastAPI(
    title="Sistema de Predição de Vendas",
    description=(
        "API para previsão de vendas utilizando "
        "AutoML, XGBoost e explicabilidade SHAP."
    ),
    version="1.0.0"
)


@app.get("/")
def inicio():
    return {
        "mensagem": "API de previsão de vendas funcionando!",
        "status": "online"
    }


@app.get("/health")
def verificar_saude():
    return {
        "status": "ok"
    }


app.include_router(previsoes_router)
app.include_router(datasets_router)
