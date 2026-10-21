
"""
API de previsão semanal (horizontes 1 a 7).

Carrega o modelo multihorizonte já treinado e
utiliza o histórico mais recente da loja.

Não modifica o modelo, o dataset original nem
os resultados da avaliação externa.
"""

import joblib
import numpy as np

from fastapi import APIRouter, HTTPException, Path

from backend.app.services.treinamento_service import (
    pasta_da_execucao,
)
from backend.app.services.atualizacao_historico_service import (
    ler_historico,
)
from backend.app.services.atributos_multihorizonte_service import (
    gerar_entradas_futuras_loja,
)
from backend.app.services.preprocessamento_service import (
    transformar_atributos,
)


router = APIRouter(
    prefix="/previsoes/execucoes",
    tags=["Previsões semanais"],
)


@router.post("/{execucao_id}/lojas/{loja_id}/semanal")
def criar_previsao_semanal(
    execucao_id: str,
    loja_id: int = Path(gt=0),
):
    """
    Prevê individualmente os próximos sete dias
    usando o modelo multihorizonte da execução.
    """

    try:
        pasta = pasta_da_execucao(execucao_id)

        caminho_modelo = (
            pasta
            / "multihorizonte"
            / "modelo_multihorizonte.joblib"
        )

        if not caminho_modelo.is_file():
            raise FileNotFoundError(
                "Esta execução ainda não possui um "
                "modelo semanal treinado."
            )

        # O artefato é produzido localmente pelo
        # pipeline do próprio sistema. Arquivos
        # joblib de origem desconhecida não são seguros.
        artefato = joblib.load(caminho_modelo)

        if (
            artefato.get("tipo") != "multihorizonte"
            or artefato.get("horizontes") != list(range(1, 8))
        ):
            raise ValueError(
                "O artefato não corresponde ao modelo "
                "multihorizonte esperado."
            )

        modelo = artefato["modelo"]
        preprocessamento = artefato["preprocessamento"]

        historico = ler_historico(pasta)

        entradas = gerar_entradas_futuras_loja(
            historico,
            loja_id,
        )

        x_futuro = transformar_atributos(
            entradas,
            preprocessamento,
        )

        valores = np.asarray(
            modelo.predict(x_futuro),
            dtype=float,
        ).reshape(-1)

        if (
            len(valores) != 7
            or not np.isfinite(valores).all()
        ):
            raise ValueError(
                "O modelo não retornou sete "
                "previsões numéricas válidas."
            )

        valores = np.maximum(valores, 0)

        previsoes = []

        for indice, (_, linha) in enumerate(
            entradas.iterrows()
        ):
            previsoes.append({
                "horizonte_dias": int(
                    linha["horizonte_dias"]
                ),
                "data_prevista": (
                    linha["data"].strftime("%Y-%m-%d")
                ),
                "vendas_previstas": round(
                    float(valores[indice]),
                    2,
                ),
            })

        if [
            item["horizonte_dias"]
            for item in previsoes
        ] != list(range(1, 8)):
            raise ValueError(
                "Os horizontes gerados estão incorretos."
            )

        total = round(
            sum(
                item["vendas_previstas"]
                for item in previsoes
            ),
            2,
        )

        data_origem = (
            entradas["data_origem"]
            .iloc[0]
            .strftime("%Y-%m-%d")
        )

        return {
            "execucao_id": pasta.name,
            "loja_id": loja_id,
            "data_origem": data_origem,
            "algoritmo": artefato["algoritmo"],
            "tipo_modelo": "piloto_multihorizonte",
            "horizonte_dias": 7,
            "previsoes": previsoes,
            "total_semanal_previsto": total,
            "observacao": (
                "As sete estimativas usam informações "
                "disponíveis até a data de origem. "
                "O modelo é um piloto multihorizonte "
                "com configurações fixas, ainda sem "
                "otimização FLAML. Promoções e outros "
                "eventos comerciais futuros não são "
                "considerados nesta versão."
            ),
        }

    except FileNotFoundError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro),
        ) from erro

    except LookupError as erro:
        raise HTTPException(
            status_code=404,
            detail=str(erro),
        ) from erro

    except ValueError as erro:
        raise HTTPException(
            status_code=422,
            detail=str(erro),
        ) from erro

    except (KeyError, TypeError) as erro:
        raise HTTPException(
            status_code=422,
            detail=(
                "O artefato do modelo semanal está "
                "incompleto ou incompatível."
            ),
        ) from erro
