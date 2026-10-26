# Marco científico — Experimentos Rossmann H7

Data: 24/10/2026
Status: experimentos centrais executados e auditados.

## Base experimental

- Dataset: Rossmann Store Sales.
- Lojas consideradas pelo AutoML: 1.115.
- Teste externo completo: 169.494 observações H7.
- Amostra científica estratificada: 63 lojas.
- Observações pareadas na amostra: 9.682.
- Origens temporais distintas: 183.
- Período das origens: 23/01/2015 a 24/07/2015.
- Horizonte: sete dias.

## Seleção AutoML

- Ferramenta: FLAML.
- Algoritmos candidatos: LightGBM e XGBoost.
- Validação temporal: quatro dobras expansivas.
- Algoritmo vencedor: XGBoost.
- RMSE de validação: 1173,8323.
- RMSE externo nas 63 lojas: 938,1865.

## Comparação externa

| Modelo | RMSE | MAE | R2 | RMSPE |
|---|---:|---:|---:|---:|
| XGBoost | 938,19 | 666,52 | 0,9212 | 14,47% |
| ARIMA | 2576,31 | 1807,52 | 0,4061 | 31,24% |
| SARIMAX | 3023,96 | 2194,16 | 0,1818 | 40,14% |
| Ingênuo sazonal | 3135,73 | 2284,22 | 0,1202 | 43,68% |

## Hipótese H1

- Redução de RMSE do XGBoost versus ARIMA: aproximadamente 63,6%.
- Teste: Diebold-Mariano unilateral.
- Diferencial: perda quadrática ARIMA menos AutoML.
- Agregação por origem temporal.
- Variância: Newey-West/Bartlett com seis defasagens.
- Estatística z: 6,9804765.
- p-valor unilateral: 1,470902742714033e-12.
- Conclusão: critérios da H1 atendidos no protocolo avaliado.

## SARIMAX

- Ajustes por loja, com seleção automática por AIC.
- Sazonalidade semanal de período 7.
- Exógenas: dow_sin, dow_cos, fim_semana.
- Promoções futuras não incluídas por falta de comprovação
  de disponibilidade antecipada.
- Primeira execução: 60 convergentes e 3 não convergentes.
- Lojas reajustadas: 59, 642 e 941.
- Limite de iterações elevado de 75 para 300.
- Resultado da auditoria final: 63/63 convergentes.
- Total preservado: 9.682 pares H7.

## Hipótese H2 — SHAP global

- Modelo: XGBoost científico previamente treinado.
- Atributos: 23.
- Observações externas alinhadas: 9.682.
- Diferença máxima entre previsões reproduzidas e FLAML:
  0,0009375.
- Linhas com NaN em media_movel_28: 41.
- Política: preservação dos NaN, sem imputação ou exclusão.
- Método: contribuições TreeSHAP.
- Subamostras: 10, com 1.008 observações cada.
- Correlações entre rankings: 45.
- Spearman médio de estabilidade: 0,9908652.
- Importância por permutação: 2.016 observações,
  três repetições.
- Spearman SHAP versus permutação: 0,8458498.
- Conclusão: ambos os critérios da H2 atendidos.

## Limitações e observações

- Resultados restritos à Rossmann e ao protocolo temporal H7.
- Subamostras SHAP podem apresentar sobreposição.
- Importância por permutação é sensível à colinearidade.
- A comparação SARIMAX utilizou exógenas de calendário.
- Os experimentos não demonstram causalidade nem garantem
  superioridade em outros datasets.

## Próximas atividades

- Conferir rankings dos atributos e gráficos para o TCC.
- Revisar redação metodológica e seção de resultados.
- Refatorar frontend com Material UI.
- Validar build e fluxo funcional.
- Registrar capturas de tela para o trabalho.

## Localização dos artefatos

- AutoML: dados/processados/experimentos/cientifico_h7_flaml/
- ARIMA: dados/processados/experimentos/baselines_h7/arima/
- SARIMAX: dados/processados/experimentos/baselines_h7/sarimax_auto/completo/
- SHAP: dados/processados/experimentos/shap_global_h7/

Os dados, checkpoints e modelos binários devem ser
preservados separadamente dos commits de código.
