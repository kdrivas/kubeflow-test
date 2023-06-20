# from aiutils import aiwr
# from aiutils.constants import c
# from aiutils import utils
from aikfp.src.automl.utils import tryconvert
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
def bq_generate_and_save(df,q,obj):   
    if (q in ['mg_metadata_constants','mg_metadata_modelo_costos'])&(obj['source']=='kf_predict'):return df
    table_name=f'tmp.{q}'.replace('mg','mg_poc')
    table_schema = []
    df.reset_index(inplace=True,drop=True)
    df['id_modelo']=obj['id_modelo']
    df['source']=obj['source']
    df['id_execution_mg']=obj['id_execution_mg']
    if q in ['mg_metadata_constants','mg_metadata_modelo_costos','mg_perf_metadata_training_validation','mg_perf_metadata_hpo_trials','mg_int_metadata_gain']:
        del df['source'],df['id_execution_mg']
        cols=['id_modelo']
    else: 
        cols=['id_modelo','id_execution_mg','source']
    # df['upload_timestamp'] = pd.Timestamp.now() - pd.Timedelta(hours=5)
    df=standard_tables(df,obj)
    if 'periodo' in df.columns: 
        try: df['periodo']=df['periodo'].apply(lambda x: x.strftime('%Y-%m-%d'))
        except: pass
        table_schema.append({'name': 'periodo', 'type': 'DATE'})
    # for x in ['sub_cols','id_features','list_of_parameters']: 
    #     if x in df.columns: table_schema.append({'name':x,'type':'ARRAY'})
    df=df[cols+list(set(df.columns).difference(cols))]
    # aiwr.save_dataset(df, 'bq://' + table_name, if_exists='append', table_schema=table_schema)  
    return df
def bq_generate_and_upload(df,q,c,d):
    df=df.rename({c:'subpoblation_features'},axis=1)
    df['periodo_execution_mg']=d
    if q=='mg_metadata_constants':
        try: del df['tuning']
        except: pass
        try: del df['training']
        except: pass
        try: del df['feature_selection']
        except: pass
    if q in ['mg_metadata_constants','mg_metadata_modelo_costos','mg_perf_metadata_training_validation','mg_perf_metadata_hpo_trials','mg_int_metadata_gain']:
        del df['periodo_execution_mg']
        cols=['id_modelo']
    else: 
        cols=['id_modelo','id_execution_mg','periodo_execution_mg','source']
    table_name=f'tmp.{q}'.replace('mg','mg_poc')
    table_schema = []
    for y in ['periodo_execution_mg','periodo']:
        if y in df.columns: 
            try: df[y]=df[y].apply(lambda x: x.strftime('%Y-%m-%d'))
            except: pass
        table_schema.append({'name': y, 'type': 'DATE'})
    if 'id_execution_mg' in df.columns: table_schema.append({'name': 'id_execution_mg', 'type': 'DATETIME'})
    print(table_schema)
    df=df[cols+list(set(df.columns).difference(cols))]
    display(df.head(3))
    aiwr.save_dataset(df, 'bq://' + table_name, if_exists='append', table_schema=table_schema) 

    
def standard_tables(df,cfg):
    try:
        display(df.head(3))
        # df['list_of_parameters']=df.apply(lambda x: x[cfg['list_of_parameters']].values,axis=1)
        # df.drop(cfg['list_of_parameters'],axis=1,inplace=True)
    except: pass
    try:
        df['sub_cols']=df.apply(lambda x: x[cfg['sub_cols']].values,axis=1)
        df.drop(cfg['sub_cols'],axis=1,inplace=True)
    except: pass
    try:
        df['id_features']=df.apply(lambda x: x[list(set(cfg['id_features']).difference(['periodo']))].values,axis=1)
        df.drop(list(set(cfg['id_features']).difference(['periodo'])),axis=1,inplace=True)
    except: pass
    try:
        df['importance']=df.apply(lambda x: x[[f'importance_{y}' for y in range(cfg[f'n_classes'])]].values,axis=1)
        df.drop([f'importance_{y}' for y in range(cfg[f'n_classes'])],axis=1,inplace=True)
    except: pass
    try:df.drop(['current_small_hist_0','current_small_hist_1','ref_small_hist_0','ref_small_hist_1'],axis=1,inplace=True)
    except: pass
    try:df.drop(['hpo','group','feature_importance'],axis=1,inplace=True)
    except: pass
    try:
        if 0 in df['name_classes'].unique():
            dc=dict(zip(range(cfg['n_classes']),cfg['sub_cols']))
            for x in cfg['sub_cols']:dc[x]=x
            df['name_classes']=df['name_classes'].map(dc)
    except: pass
    df.rename(dict(zip(cfg['kpi_features'],['kpi_1','kpi_2'])),axis=1,inplace=True)
    try:
        df['kpi_1']=df['kpi_1'].astype(float)
        df['kpi_2']=df['kpi_2'].astype(float)
    except: pass
    df.rename({cfg['name_target']:'name_target'},axis=1,inplace=True)
    df.rename({'target':'name_target'},axis=1,inplace=True)
    # df.columns.name='name_target'
    df.rename({cfg['subpoblation_features'][0]:'subpoblation_features'},axis=1,inplace=True)
    return df
def create_leaderboard_dataframe(mg):
    rfp=mg['perf__general_metrics']
    rfd=mg['feature__drift']['evidently']['_general_drift']
    rfi=mg['feature__explainer']['feature__explainer_general']
    display(rfi.head(3))
    return pd.DataFrame({
                'log_loss':rfp[(rfp['metric']=='log_loss')&(rfp['type_metric']=='performance_by_all')]['value'].values[0],
                'accuracy':rfp[(rfp['metric']=='accuracy')&(rfp['type_metric']=='performance_by_all')]['value'].values[0],
                'roc_auc_score':rfp[(rfp['metric']=='roc_auc_score')&(rfp['type_metric']=='performance_by_all')]['value'].values[0],
                'gini':rfp[(rfp['metric']=='gini')&(rfp['type_metric']=='performance_by_all')]['value'].values[0],
                'precision':rfp[(rfp['metric']=='precision')&(rfp['type_metric']=='performance_by_all')]['value'].mean(),
                'recall':rfp[(rfp['metric']=='recall')&(rfp['type_metric']=='performance_by_all')]['value'].mean(),
                'f1_score':rfp[(rfp['metric']=='f1_score')&(rfp['type_metric']=='performance_by_all')]['value'].mean(),
                 'fi_shap_0':(rfi[(rfi['type_metric']=='explainer_by_all')&(rfi['metric']=='shap_abs')&(rfi['name_classes']=='all')]['value']>0).mean(),
        'fi_shap_0_5':(rfi[(rfi['type_metric']=='explainer_by_all')&(rfi['metric']=='shap_abs')&(rfi['name_classes']=='all')]['value']>0.5).mean(),
                'feature_drift': rfd['share_drifted_features'].values[0],
                 'target_drift': rfd['real_drift'].values[0],
                 'prediction_drift': rfd['prediction_drift'].values[0],
                # 'color_mae': tryconvert(mg,np.NAN, lambda x:min([mean_absolute_error(rfp[rfp['metric']==f"target_rlt"&(rf['percentil']!='all')]['value'],
                #                                                                      rfp[rfp['metric']==f"prediction_rlt"&(rf['percentil']!='all')]['value']) for x in _rf])),
                 'kpi_mae': tryconvert(rfp,np.NAN, lambda rfp:mean_absolute_error(rfp[(rfp['metric']==f"target_rlt")]['value'],
                                                                                 rfp[rfp['metric']==f"prediction_rlt"]['value']))                                     ,

                'cpu_usage_percent':tryconvert(mg,np.NAN,lambda mg: mg['metadata']['metadata__costs'][['cpu_usage_m_1','cpu_usage_0','cpu_usage_s_1']].mean(axis=1).mean(axis=0)),
                'ram_usage_percent':tryconvert(mg,np.NAN,lambda mg: (mg['metadata']['metadata__costs']['memory_usage_available']/mg['metadata']['metadata__costs']['memory_usage_total']).mean())
    },index=[0])