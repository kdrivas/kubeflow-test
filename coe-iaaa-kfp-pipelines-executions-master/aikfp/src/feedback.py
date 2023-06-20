import pandas as pd
import numpy as np
from .automl import model_outputs as Z
# from aiutils import aiwr
def get_feedback(df,df_test,obj,preprocess,model):
    df['set_analysis']='training'
    df_test['set_analysis']='predict'
    df=pd.concat([df.reset_index(drop=True),df_test.reset_index(drop=True)])
    dict_target=dict(zip(obj['sub_cols'],range(obj['n_classes'])))
    df.reset_index(inplace=True,drop=True)
    try:
        yv=Z.cls_predict_proba(model,preprocess.transform(df)[Z.cls_feature_name(model,obj['type_model'])],obj['type_model'],obj['cat_features'])
    except:
        yv=Z.cls_predict_proba(model,pd.DataFrame(preprocess.transform(df),columns=obj['name_features'])[Z.cls_feature_name(model,obj['type_model'])],obj['type_model'],obj['cat_features'])
    for n,x in enumerate(obj['sub_cols']): 
        df[x]=yv[:,n]
    df[obj['name_prediction']]=df[obj['sub_cols']].idxmax(axis=1).map(dict_target)
    for n,x in enumerate(obj['sub_cols']):
        df[f'{obj["name_prediction"]}_{n}']=df[obj['name_prediction']].apply(lambda y: 1 if y==n else 0)
        df[f'{obj["name_target"]}_{n}']=df[obj['name_target']].apply(lambda y: 1 if y==n else 0)
        df[f'{x}_cut']=pd.cut(df[x],10,labels=['P010','P020','P030','P040','P050','P060','P070','P080','P090','P100'])
    l=[x for x in df.columns if x in [obj['name_target']]+[obj['name_prediction']]+obj['sub_cols']+['set_analysis']+[obj['kpi_features']]+[obj['subpoblation_features']]+[obj['evol_features']]+obj['text_features']]
    l.extend([x for x in df.columns if obj['name_target'] in x])
    l.extend([x for x in df.columns if obj['name_prediction'] in x])
    for y in obj['sub_cols']:l.extend([x for x in df.columns if y in x])
    df=df[obj['id_features']+obj['name_features']+l]    
    df=df.loc[:,~df.columns.duplicated()]
    df.reset_index(inplace=True,drop=True)
    # df=df[obj['id_features']+obj['cat_features']+obj['cont_features']+l]
    return df
def add_period_in_dataframe(path,period,df_id,cfg,type_='parquet'):
    df_id['periodo']=df_id['periodo'].apply(lambda x: str(x)[:10])
    if type_=='parquet':mf=pd.read_parquet(path)
    elif type_=='aiwr':mf=aiwr.read_dataset(path+' LIMIT 50000')
    mf['periodo']=period    
    if len(mf.merge(df_id,on=cfg))!=0:mf=mf.merge(df_id,on=cfg)
    else: 
        mf=mf.merge(df_id[[x for x in df_id.columns if 'periodo' not in x]],on=cfg[0])
    return mf
