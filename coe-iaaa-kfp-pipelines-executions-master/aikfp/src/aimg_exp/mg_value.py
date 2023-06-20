from sklearn.metrics import precision_score,recall_score,accuracy_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from fairlearn.metrics import MetricFrame,count,selection_rate
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import shap
import graphviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def generate_color_analysis(xf,cfg):
    xf=xf[xf['set_analysis']=='predict']
    wf=[]
    y='cut'
    for x in cfg['sub_cols']:
        rf=xf.groupby(f'{x}_{y}').agg({cfg['name_target']:['count','mean','sum'],x:['mean','sum']}).reset_index()
        rf.columns=[f'percentil','count','target_rlt','target_abs','prediction_rlt','prediction_abs']
        rf['name_classes']=x
        # rf['bin']=y
        rf=pd.melt(rf,id_vars=['percentil','name_classes'],value_vars=['count','target_rlt','target_abs','prediction_rlt','prediction_abs'],var_name='metric')
        rf[cfg['subpoblation_features']]='all'
        rf['cutoff']='all'
        rf['type_metric']='captura_by_all'
        wf.append(rf)
    return pd.concat(wf)
def generate_color_feature_analysis(xf,cfg):
    xf=xf[xf['set_analysis']=='predict']
    wf=[]
    y='cut'
    for x in cfg['sub_cols']:
        for w in [cfg['name_target'],cfg['name_prediction']]:
            for f in cfg['cat_features']+cfg['cont_features']:
                try:
                    if f in cfg['cont_features']:
                        xf[f]=pd.cut(xf[f],10,labels=['P010','P020','P030','P040','P050','P060','P070','P080','P090','P100'])
                        rf=xf.copy()
                    elif f in cfg['cat_features']:
                        if len(xf[f].value_counts())>=10: df=xf[xf[f].isin(xf[f].value_counts().index.tolist()[:10])]  
                        else: rf=xf.copy()
                    rf=pd.crosstab(rf[f],rf[f'{x}_{y}'],normalize='index').reset_index()
                    rf.index=range(len(rf))
                    rf.index.name = None
                    rf['feature']=f
                    rf.rename({f:'group_feature'},axis=1,inplace=True)
                    rf['group_feature']=rf['group_feature'].astype(str)
                    rf['name_classes']=x
                    rf['flg_target']=w
                    # rf['bin']=y
                    p_columns=[x for x in rf if (x[0]=='P')&(x[-1]=='0')]
                    
                    rf=pd.melt(rf,id_vars=['feature','group_feature','name_classes','flg_target'],value_vars=p_columns,var_name=f'metric')
                    rf['type_metric']='report_detail'
                    wf.append(rf)
                except: pass
    return wf