
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.previsao_service import prever_vendas
from backend.app.services.explicabilidade_service import explicar_previsao


router = APIRouter(
    prefix="/previsoes",
    tags=["Previsões"]
)


class AtributosPrevisao(BaseModel):
    promocao: float
    promocao_continua: float
    mes_renovacao_promocao: float | None = None
    feriado: float
    feriado_escolar: float
    distancia_concorrente: float | None = None

    dia_semana: int = Field(ge=1, le=7)
    dia_mes: int = Field(ge=1, le=31)
    mes: int = Field(ge=1, le=12)
    semana_ano: int = Field(ge=1, le=53)

    lag_7: float | None = None
    lag_14: float | None = None
    lag_21: float | None = None
    lag_28: float | None = None
    media_movel_7: float | None = None
    media_movel_28: float | None = None

    tipo_loja: str
    sortimento: str


class SolicitacaoPrevisao(BaseModel):
    loja_id: int = Field(gt=0)
    data_origem: date
    data: date
    atributos: AtributosPrevisao


def validar_horizonte(solicitacao: SolicitacaoPrevisao):
    """Valida o horizonte fixo de sete dias."""
    if solicitacao.data != (
        solicitacao.data_origem + timedelta(days=7)
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "A data prevista deve estar exatamente "
                "7 dias após a data de origem."
            )
        )


@router.post("/")
def criar_previsao(solicitacao: SolicitacaoPrevisao):
    """Gera uma previsão utilizando o XGBoost."""

    validar_horizonte(solicitacao)

    try:
        previsao = prever_vendas(
            solicitacao.atributos.model_dump()
        )[0]

        return {
            "loja_id": solicitacao.loja_id,
            "data_origem": solicitacao.data_origem,
            "data_prevista": solicitacao.data,
            "horizonte_dias": 7,
            "vendas_previstas": round(previsao, 2),
            "algoritmo": "XGBoost"
        }

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro


@router.post("/explicacao")
def criar_previsao_com_explicacao(
    solicitacao: SolicitacaoPrevisao
):
    """Gera a previsão e suas contribuições SHAP."""

    validar_horizonte(solicitacao)

    try:
        resultado = explicar_previsao(
            solicitacao.atributos.model_dump(),
            quantidade_atributos=10
        )

        return {
            "loja_id": solicitacao.loja_id,
            "data_origem": solicitacao.data_origem,
            "data_prevista": solicitacao.data,
            "horizonte_dias": 7,
            "algoritmo": resultado["algoritmo"],
            "vendas_previstas": resultado["previsao"],
            "explicabilidade": {
                "metodo": "SHAP",
                "valor_base": resultado["valor_base_shap"],
                "principais_contribuicoes": (
                    resultado["principais_contribuicoes"]
                ),
                "diferenca_reconstrucao": (
                    resultado["diferenca_reconstrucao"]
                ),
                "observacao": resultado["observacao"]
            }
        }

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro)
        ) from erro
