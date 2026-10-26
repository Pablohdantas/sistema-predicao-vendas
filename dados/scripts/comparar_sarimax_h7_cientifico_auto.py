"""Baseline SARIMAX H7 pareado ao FLAML Rossmann (63 lojas).

Primeiro executar --modo verificar; piloto: --modo executar --max-lojas 1 --max-origens 5.
Completo: --modo executar (sem limites). Os resultados de piloto ficam em pasta separada.

Ordens (p,d,q)(P,D,Q,7) selecionadas automaticamente por pmdarima com AIC,
exclusivamente no histórico de treinamento de cada loja.

Covariáveis padrão estritamente conhecidas no instante da origem: dia da semana
(seno/cosseno) e fim de semana. Promoções futuras NÃO são usadas, pois o dataset
histórico isolado não demonstra que a agenda futura estava disponível naquele dia.
Assim, este SARIMAX é sazonal com exógenas de calendário; se o TG exigir promoções,
será preciso demonstrar disponibilidade ex ante e alterar o protocolo antes de
considerar esta execução definitiva.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DADOS = ROOT / 'dados'
FONTE = DADOS / 'processados/experimentos/cientifico_h7_flaml/previsoes_teste_h7_63_lojas.csv'
AMOSTRA = DADOS / 'processados/amostra_arima.csv'
RAW = DADOS / 'brutos/train.csv'
SAIDA = DADOS / 'processados/experimentos/baselines_h7/sarimax_auto'
CORTE = pd.Timestamp('2015-01-23')
COLS = ['loja_id','data_origem','data','vendas','previsao_automl','previsao_sarimax']


def salvar_csv(df, destino):
    destino.parent.mkdir(parents=True,exist_ok=True)
    tmp = destino.with_name(destino.name+'.tmp')
    df.to_csv(tmp,index=False,date_format='%Y-%m-%d')
    os.replace(tmp,destino)


def salvar_json(obj,destino):
    destino.parent.mkdir(parents=True,exist_ok=True)
    tmp=destino.with_name(destino.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,destino)


def metricas(y,p):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    err=y-p; var=np.sum((y-np.mean(y))**2); pos=y>0
    return {'quantidade':int(len(y)),'rmse':float(np.sqrt(np.mean(err**2))),
            'mae':float(np.mean(np.abs(err))),
            'r2':float(1-np.sum(err**2)/var) if var>0 else None,
            'rmspe':float(np.sqrt(np.mean((err[pos]/y[pos])**2))) if pos.any() else None}


def previsoes():
    p=pd.read_csv(FONTE,parse_dates=['data_origem','data'])
    ids=pd.read_csv(AMOSTRA).loja_id.astype(int).tolist()
    required={'loja_id','data_origem','data','horizonte_dias','vendas','vendas_previstas'}
    if not required.issubset(p.columns):raise ValueError('Colunas faltando na fonte FLAML')
    if len(ids)!=63 or len(set(ids))!=63 or len(p)!=9682 or p.loja_id.nunique()!=63 or set(p.loja_id)!=set(ids):
        raise ValueError('Amostra FLAML não corresponde a 63 lojas / 9682 pares')
    if p.duplicated(['loja_id','data_origem','data']).any():raise ValueError('Pares duplicados')
    if not (p.horizonte_dias==7).all() or not (p.data-p.data_origem==pd.Timedelta(days=7)).all():
        raise ValueError('Horizonte inconsistente')
    if p.data_origem.min()<CORTE or not np.isfinite(p[['vendas','vendas_previstas']].to_numpy()).all():
        raise ValueError('Valores inválidos')
    return p.sort_values(['loja_id','data_origem']).reset_index(drop=True)


def historico_bruto(ids):
    r=pd.read_csv(RAW,usecols=['Store','Date','Sales','Open'],parse_dates=['Date'])
    r=r[r.Store.isin(ids)].copy()
    if r.duplicated(['Store','Date']).any():raise ValueError('Datas duplicadas no bruto')
    r.loc[r.Open==0,'Sales']=0.0
    return r


def regularizar(df,ultima_origem):
    """Imputação causal: somente os valores anteriores ao dia ausente."""
    datas=pd.date_range(df.index.min(),ultima_origem,freq='D')
    src=df.Sales.reindex(datas)
    por_dia={i:[] for i in range(7)}; total=[]; valores=[]; faltas=0
    for dia,valor in src.items():
        if pd.isna(valor):
            faltas+=1
            ref=por_dia[dia.dayofweek] or total
            valores.append(float(np.median(ref)) if ref else 0.0)
        else:
            valor=float(valor); valores.append(valor)
            por_dia[dia.dayofweek].append(valor); total.append(valor)
    return pd.Series(valores,index=datas,dtype=float),faltas


def exogenas(datas):
    """Exógenas obtidas só do calendário; disponíveis para datas futuras."""
    datas=pd.DatetimeIndex(datas)
    dow=datas.dayofweek.to_numpy()
    return pd.DataFrame({'dow_sin':np.sin(2*np.pi*dow/7),
                         'dow_cos':np.cos(2*np.pi*dow/7),
                         'fim_semana':(dow>=5).astype(float)},index=datas)


def validar_checkpoint(c,esperado):
    if list(c.columns)!=COLS or len(c)!=len(esperado):raise ValueError('Checkpoint inválido')
    c=c.sort_values(['loja_id','data_origem','data']).reset_index(drop=True)
    e=esperado.sort_values(['loja_id','data_origem','data']).reset_index(drop=True)
    for col in ['loja_id','data_origem','data']:
        if not c[col].equals(e[col]):raise ValueError('Checkpoint com chaves divergentes')
    if not np.allclose(c.vendas,e.vendas,atol=1e-8,rtol=0) or not np.allclose(c.previsao_automl,e.vendas_previstas,atol=1e-5,rtol=0):
        raise ValueError('Checkpoint divergente da fonte')
    if not np.isfinite(c.previsao_sarimax).all():raise ValueError('Previsões inválidas')
    return c


def executar(max_lojas,max_origens,retomar,maxiter):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from pmdarima import auto_arima
    p=previsoes()
    grupos=list(p.groupby('loja_id',sort=True))
    if max_lojas is not None:grupos=grupos[:max_lojas]
    piloto=max_lojas is not None or max_origens is not None
    saida=SAIDA / ('piloto_'+str(max_lojas or 63)+'lojas_'+str(max_origens or 'todas')+'origens' if piloto else 'completo')
    ck=saida/'lojas';ck.mkdir(parents=True,exist_ok=True)
    raw=historico_bruto([int(loja) for loja,_ in grupos])
    saidas=[];ajustes=[]
    for numero,(loja,grupo) in enumerate(grupos,1):
        loja=int(loja)
        e=grupo.sort_values('data_origem').reset_index(drop=True)
        if max_origens is not None:e=e.head(max_origens).copy()
        arquivo=ck/f'loja_{loja}.csv';detalhe=ck/f'loja_{loja}.json'
        if (
            retomar
            and arquivo.exists()
            and detalhe.exists()
            and json.loads(detalhe.read_text(encoding="utf-8")).get("convergiu") is True
        ):
            c=validar_checkpoint(pd.read_csv(arquivo,parse_dates=['data_origem','data']),e)
            saidas.append(c);ajustes.append(json.loads(detalhe.read_text(encoding='utf-8')))
            print(f'[{numero}/{len(grupos)}] Loja {loja}: checkpoint recuperado',flush=True)
            continue
        loja_raw=raw[raw.Store==loja].set_index('Date').sort_index()
        if loja_raw.empty:raise ValueError(f'Loja {loja} sem série')
        serie,lacunas=regularizar(loja_raw,e.data_origem.max())
        treino=serie.loc[:CORTE]
        if len(treino)<180:raise ValueError(f'Loja {loja}: treino insuficiente')
        x_inicial = exogenas(treino.index).to_numpy(dtype=float)
        # Apenas treino ate o corte. AIC define ordens, nunca usa o teste externo.
        seletor = auto_arima(
            treino.to_numpy(dtype=float), X=x_inicial,
            seasonal=True, m=7,
            start_p=0, start_q=0, max_p=2, max_q=2, max_d=1,
            start_P=0, start_Q=0, max_P=1, max_Q=1, max_D=1,
            max_order=4, stepwise=True, information_criterion='aic',
            suppress_warnings=True, error_action='ignore', trace=False,
            maxiter=maxiter, with_intercept='auto',
        )
        ordem = tuple(int(v) for v in seletor.order)
        ordem_sazonal = tuple(int(v) for v in seletor.seasonal_order)
        if len(ordem) != 3 or len(ordem_sazonal) != 4 or ordem_sazonal[3] != 7:
            raise ValueError(f'Loja {loja}: ordens sazonais invalidas')
        # Reajuste com mesma estrutura de ordens e exogenas para atualizacao statespace.
        ajuste=SARIMAX(treino.to_numpy(dtype=float),exog=x_inicial,
                       order=ordem,seasonal_order=ordem_sazonal,trend='c',
                       enforce_stationarity=False,enforce_invertibility=False)
        resultado=ajuste.fit(disp=False,maxiter=maxiter)
        if not np.isfinite(resultado.params).all():raise ValueError(f'Loja {loja}: parâmetros não finitos')
        convergiu = bool(resultado.mle_retvals.get('converged', False))
        if not convergiu:
            print(f'AVISO: loja {loja} com otimizacao statsmodels sem convergencia confirmada',flush=True)
        origem_anterior=CORTE; registros=[]
        for linha in e.itertuples(index=False):
            origem=linha.data_origem
            if origem>origem_anterior:
                dias=pd.date_range(origem_anterior+pd.Timedelta(days=1),origem,freq='D')
                novos=serie.reindex(dias)
                if novos.isna().any():raise ValueError('Calendário irregular após regularização')
                resultado=resultado.append(novos.to_numpy(),exog=exogenas(dias).to_numpy(),refit=False)
                origem_anterior=origem
            futuras=pd.date_range(origem+pd.Timedelta(days=1),periods=7,freq='D')
            pred=float(resultado.forecast(steps=7,exog=exogenas(futuras).to_numpy())[-1])
            if not np.isfinite(pred):raise ValueError(f'Loja {loja}: previsão não finita em {origem.date()}')
            registros.append({'loja_id':loja,'data_origem':origem,'data':linha.data,
                              'vendas':float(linha.vendas),'previsao_automl':float(linha.vendas_previstas),
                              'previsao_sarimax':max(0.0,pred)})
        c=pd.DataFrame(registros,columns=COLS)
        detalhes={'loja_id':loja,'ordem':list(ordem),'ordem_sazonal':list(ordem_sazonal),
                  'convergiu':convergiu,'selecao_ordens':'pmdarima.auto_arima, AIC, somente treino',
                  'n_treino_inicial':len(treino),'pares':len(c),'lacunas':lacunas,
                  'atualizacao':'statespace append(refit=False)'}
        salvar_csv(c,arquivo);salvar_json(detalhes,detalhe)
        saidas.append(c);ajustes.append(detalhes)
        print(f'[{numero}/{len(grupos)}] Loja {loja}: SARIMAX{ordem}x{ordem_sazonal}, {len(c)} pares',flush=True)
    todos=pd.concat(saidas,ignore_index=True).sort_values(['data_origem','loja_id']).reset_index(drop=True)
    if not piloto:
        if len(todos)!=9682 or todos.loja_id.nunique()!=63:raise ValueError('Cobertura incompleta')
    y=todos.vendas.to_numpy()
    resumo={'tipo':'sarimax_h7_piloto' if piloto else 'sarimax_h7_completo',
            'covariaveis':['dow_sin','dow_cos','fim_semana'],
            'nota':'Promoções futuras excluídas por falta de prova de disponibilidade ex ante; verificar alinhamento ao TG.',
            'selecao_ordens':'auto_arima por loja sobre treino, m=7, AIC',
            'lojas':int(todos.loja_id.nunique()),'pares':len(todos),
            'origens':int(todos.data_origem.nunique()),
            'automl':metricas(y,todos.previsao_automl),'sarimax':metricas(y,todos.previsao_sarimax),
            'ajustes':ajustes,'gerado_em_utc':datetime.now(timezone.utc).isoformat()}
    salvar_csv(todos,saida/'comparacao_sarimax_h7.csv')
    salvar_json(resumo,saida/'resumo_sarimax_h7.json')
    print(json.dumps({k:v for k,v in resumo.items() if k!='ajustes'},indent=2,ensure_ascii=False))
    print('Artefatos:',saida)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--modo',choices=['verificar','executar'],default='verificar')
    ap.add_argument('--max-lojas',type=int)
    ap.add_argument('--max-origens',type=int)
    ap.add_argument('--sem-retomar',action='store_true')
    ap.add_argument('--maxiter',type=int,default=75)
    args=ap.parse_args()
    p=previsoes()
    print(json.dumps({'status':'fonte_verificada','pares':len(p),'lojas':p.loja_id.nunique(),
                      'origens':p.data_origem.nunique()},indent=2))
    if args.modo=='executar':
        if args.max_lojas is not None and not 1<=args.max_lojas<=63:raise ValueError('--max-lojas: 1..63')
        if args.max_origens is not None and not 1<=args.max_origens<=183:raise ValueError('--max-origens: 1..183')
        executar(args.max_lojas,args.max_origens,not args.sem_retomar,args.maxiter)

if __name__=='__main__':main()
