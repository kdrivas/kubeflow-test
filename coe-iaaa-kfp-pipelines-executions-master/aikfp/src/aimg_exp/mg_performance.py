import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns       
import lightgbm as lgb
from datetime import datetime
from fairlearn.metrics import (
    MetricFrame,
    count,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate,
    selection_rate,
)
from sklearn.metrics import log_loss,precision_score,recall_score,accuracy_score,f1_score,roc_auc_score,confusion_matrix,classification_report,roc_curve,auc,precision_recall_curve
import numpy as np
import optuna
from aikfp.src.automl import utils
from aikfp.src.automl import model_outputs as Z
from . import querys as Q
from . import mg_value as V
def exp_perf_general_metric(df,name_target,name_prediction,sub_cols,
                    features,
                    kpi_features,
                    _vars): 
    _mf,_cf,_nf=[],[],[]       
    def report_general(df,_mf,_cf,mode='all',fmode='all'):
        mf=pd.DataFrame({'metric':['log_loss','accuracy','roc_auc_score','gini','count'],
                      'value':[utils.tryconvert(df,np.NAN,lambda df: log_loss(df[name_target],df[sub_cols].values)) ,
                               utils.tryconvert(df,np.NAN,lambda df: accuracy_score(df[name_target], df[name_prediction])) ,                                         utils.tryconvert(df,np.NAN,lambda df: Z.cls_metric_auc(df[name_target],df[sub_cols].values))  ,
                                     2*utils.tryconvert(df,np.NAN,lambda df: Z.cls_metric_auc(df[name_target],df[sub_cols].values))-1 ,
                               count(df[name_target],df[sub_cols].values),
                                     ]})
        df[f'kpi_w_target']=df[kpi_features]*df[name_target]
        df[f'kpi_w_prediction']=df[kpi_features]*df[name_prediction]
        df=df.rename({name_target:'target',name_prediction:'prediction'},axis=1)
        f=['kpi_w_target','kpi_w_prediction','target','prediction']
        cf=pd.concat([df[f].mean().rename(dict(zip(f,[f'{x}_rlt' for x in f])),axis=0),
                      df[f].sum().rename(dict(zip(f,[f'{x}_abs' for x in f])),axis=0)],
                     axis=0).reset_index().rename({'index':'metric',0:'value'},axis=1)
        cf['type_metric']=f'value_by_{mode}'
        mf['type_metric']=f'performance_by_{mode}'
        cf['name_classes']='all'
        mf['name_classes']='all'
        cf['percentil']='all'
        mf['percentil']='all'
        cf['cutoff']=100
        mf['cutoff']=100
        cf[features]=fmode
        mf[features]=fmode
        _mf.append(mf);_cf.append(cf)
        return _mf,_cf
    def report_particular(df,_mf,mode='all',fmode='all'):
        try:
            mf=pd.concat([pd.DataFrame({
                    'name_classes':x,
                    'metric':['recall','precision','f1_score','false_positive_rate','false_negative_rate','true_positive_rate','true_negative_rate','selection_rate'],
                          'value':[ recall_score(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0)),
                                  precision_score(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0)),
                                  f1_score(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0)),
                                  false_positive_rate(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0)),
                                   false_negative_rate(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0)),
                                true_positive_rate(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0)),
                                   true_negative_rate(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0)),
                                   selection_rate(np.where(df[name_target]==x,1,0), np.where(df[name_prediction]==x,1,0))
                                         ]}) for x in df[name_prediction].unique()])
            mf['type_metric']=f'performance_by_{mode}'
            mf[features]=fmode
            mf['cutoff']=100
            mf['percentil']='all'
            _mf.append(mf)
            return _mf
        except: return _mf

    _mf,_cf=report_general(df,_mf,_cf,'all','all')
    _nf=report_particular(df,_nf,'all','all')
    for x in [x for x in _vars if x!='nan']: 
        _mf,_cf=report_general(df[df[features]==x],_mf,_cf,'subpoblation',x)
        _nf=report_particular(df[df[features]==x],_nf,'subpoblation',x)
    _rf=exp_perf_particular_cutoff_metric(df,name_target,name_prediction,sub_cols,
                    features,
                    kpi_features,
                    _vars)
    _zf=V.generate_color_analysis(df,{'sub_cols':sub_cols,'name_target':name_target,'subpoblation_features':features})
    return pd.concat([pd.concat(_nf),pd.concat(_mf),pd.concat(_cf),_rf,_zf])
def exp_perf_particular_cutoff_metric(df,name_target,name_prediction,sub_cols,
                    features,
                    kpi_features,
                    _vars):
    _mf=[]
    for n,z in enumerate(sub_cols):
        def report(df,_mf,mode):
            yr=df[name_target].apply(lambda x: 1 if x==n else 0)
            rf=pd.DataFrame([{'cutoff':x,'name_classes':z,
            'precision':precision_score(yr,df[z].apply(lambda y: 1 if y>x else 0)),
            'count':count(yr,df[z].apply(lambda y: 1 if y>x else 0)),
            'recall':recall_score(yr,df[z].apply(lambda y: 1 if y>x else 0)),
            'f1_score':f1_score(yr,df[z].apply(lambda y: 1 if y>x else 0)),
            'true_positive_rate':utils.tryconvert(df,np.NAN,lambda df: true_positive_rate(yr,df[z].apply(lambda y: 1 if y>x else 0))),
            'false_positive_rate':utils.tryconvert(df,np.NAN,lambda df: false_positive_rate(yr,df[z].apply(lambda y: 1 if y>x else 0))),
            'true_negative_rate':utils.tryconvert(df,np.NAN,lambda df: true_negative_rate(yr,df[z].apply(lambda y: 1 if y>x else 0))),
            'false_negative_rate':utils.tryconvert(df,np.NAN,lambda df: false_negative_rate(yr,df[z].apply(lambda y: 1 if y>x else 0))),
                    'selection_rate':utils.tryconvert(df,np.NAN,lambda df: selection_rate(yr,df[z].apply(lambda y: 1 if y>x else 0))),
                             } for x in np.linspace(0,1,50)] )
            mf=pd.melt(rf,id_vars=['cutoff','name_classes'],value_vars=['recall','precision','f1_score','false_positive_rate','false_negative_rate','true_positive_rate','true_negative_rate','selection_rate'],var_name='metric')
            mf['type_metric']=f'performance_w_cutoff_by_{mode}'
            mf[features]='all'
            mf['percentil']='all'
            _mf.append(mf)
            return _mf
        _mf=report(df,_mf,'all')
        for x in [x for x in _vars if x!='nan']: _mf=report(df[df[features]==x],_mf,'subpoblation')
        
    return pd.concat(_mf)

def perf__hpo_trials(rf):
    rf=rf.trials_dataframe()
    list_of_parameters=[x for x in rf.columns if 'params_' in x if rf[x].dtypes in ['int','float']]
    rf=rf[['number','value','state']+list_of_parameters].rename({'number':'trial','value':'value_trial','state':'state_trial'},axis=1)
    return pd.melt(rf,id_vars=['trial','state_trial','value_trial'],value_vars=list_of_parameters,var_name='parameter')