from . import querys as Q
import pandas as pd
import numpy as np
import json
from evidently.dashboard import Dashboard
from evidently.model_profile import Profile
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import (
    DataDriftTab,
    CatTargetDriftTab,
    NumTargetDriftTab,
    RegressionPerformanceTab,
    ClassificationPerformanceTab,
    ProbClassificationPerformanceTab,
    DataQualityTab
)
from evidently.model_profile.sections import (
    DataDriftProfileSection,
    CatTargetDriftProfileSection,
    NumTargetDriftProfileSection,
    RegressionPerformanceProfileSection,
    ClassificationPerformanceProfileSection,
    ProbClassificationPerformanceProfileSection,
    DataQualityProfileSection
)
from alibi_detect.cd import KSDrift
from .mg_report import st_bars
from aikfp.src.automl import utils
from .mg_value import generate_color_feature_analysis
def exp_feature_drift(df,x_train,x_test,y_train,y_test,preprocess,obj):
    _xf,xf=x_train[obj['cat_features']+obj['cont_features']+[obj['name_prediction']]],x_test[obj['cat_features']+obj['cont_features']+[obj['name_prediction']]]
    _xf,xf,column_mapping=evidently_samples(_xf,y_train,
                                    xf,y_test,obj['sub_cols'],
                                    obj['name_target'],obj['name_prediction'],obj['cont_features'],obj['cat_features'],obj['evol_features'])
    # evidently_dashboard(_xf,xf,column_mapping,{id_execution},mg_output)
    obj_=evidently_profile(_xf,xf,column_mapping)
    return {'data_distributions':Q.bq_generate_and_save(data_distribution_get_objects(df,obj),
                                                    'mg_feature__distribution',obj
                                                    ),
            'evidently':evidently_get_objects(_xf,xf,obj_,obj['cat_features'],obj['cont_features'],obj,preprocess),
            }
def evidently_samples(x_train,y_train,x_test,y_test,sub_cols,name_target,name_prediction,numerical_features,categorical_features,datetime_features):
    _xf=pd.concat([x_train.reset_index(drop=True),
                  pd.DataFrame(y_train).reset_index(drop=True)],axis=1)
    xf=pd.concat([x_test.reset_index(drop=True),
                  pd.DataFrame(y_test).reset_index(drop=True)],axis=1)
    if len(_xf)>=5000:_xf=_xf.sample(5000,random_state=60)
    if len(xf)>=5000:xf=xf.sample(5000,random_state=60)
    column_mapping = ColumnMapping()
    column_mapping.target = name_target
    column_mapping.target_names = name_target
    column_mapping.prediction = name_prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    column_mapping.datetime = datetime_features
    column_mapping.confidence=0.05
    column_mapping.threshold=0.01
    return _xf,xf,column_mapping
def evidently_profile(_xf,xf,column_mapping):
    try:
        profile = Profile(sections=[
            CatTargetDriftProfileSection(),
            DataDriftProfileSection(),                        
            DataQualityProfileSection()
                                   ])
        profile.calculate(_xf,xf, column_mapping = column_mapping)
    except:
        profile = Profile(sections=[
            # CatTargetDriftProfileSection(),
            DataDriftProfileSection(),                        
            DataQualityProfileSection()
                               ])
        profile.calculate(_xf,xf, column_mapping = column_mapping)
    return json.loads(profile.json())
            
def evidently_get_objects(_xf,xf,obj_,categorical_features,numerical_features,metadata,preprocess):    
    af=alibi_detect_get_objects(_xf,xf,categorical_features,numerical_features,metadata,preprocess)
    wf=create_specific_drift(obj_)
    wf=wf.merge(af,how='left',on='feature')
    drift_columns=list(set(wf.columns).difference(['feature']))
    wf=pd.melt(wf,id_vars=['feature'],value_vars=drift_columns,var_name='metric')
    wf['type_metric']=wf['metric'].apply(lambda x: f'drift_indicator' if ('hist_ref_' not in x)&('hist_current_' not in x) else 'drift_bins')
    return {
            '_general_drift':pd.DataFrame({'n_features':obj_['data_drift']['data']['metrics']['n_features'],
                        'n_drifted_features':obj_['data_drift']['data']['metrics']['n_drifted_features'],
                        'share_drifted_features':obj_['data_drift']['data']['metrics']['share_drifted_features'],
                        'dataset_drift':obj_['data_drift']['data']['metrics']['dataset_drift'],
                        'prediction_drift':utils.tryconvert(obj_,np.NAN, lambda y: y['cat_target_drift']['data']['metrics']['prediction_drift']), 
                            'real_drift':utils.tryconvert(obj_,np.NAN, lambda y: y['cat_target_drift']['data']['metrics']['target_drift']), 
            },index=[0]),
        '_specific_drift':Q.bq_generate_and_save(wf,'mg_features__drift_specific',metadata),
               }
def alibi_detect_get_objects(_xf,xf,categorical_features,numerical_features,metadata,preprocess):
    _xf=pd.DataFrame(preprocess.transform(_xf),columns=metadata['name_features'])
    xf=pd.DataFrame(preprocess.transform(xf),columns=metadata['name_features'])
    cd = KSDrift(_xf[categorical_features+numerical_features].values, p_val=0.01)
    preds = cd.predict(xf[categorical_features+numerical_features].values, drift_type='batch', return_p_val=True, return_distance=True)
    return pd.DataFrame({'feature':categorical_features+numerical_features,'distance':preds['data']['distance'],
                                                'p_val_distance':preds['data']['p_val']}).sort_values('distance',ascending=False)
    
def evidently_dashboard(_xfs,xfs,column_mapping,name,mg_output):
    try:
        d=Dashboard(tabs=[DataQualityTab(verbose_level=1),
                         DataDriftTab(verbose_level=1),
                          CatTargetDriftTab(verbose_level=1),
                          ClassificationPerformanceTab(verbose_level=1)
                         ])
        d.calculate(_xf,xf, column_mapping = column_mapping)
    except: 
        d=Dashboard(tabs=[DataQualityTab(verbose_level=1),
                         DataDriftTab(verbose_level=1),
                          # CatTargetDriftTab(),
                          ClassificationPerformanceTab(verbose_level=1)
                         ])
        d.calculate(_xf,xf, column_mapping = column_mapping)
    d.save(f"{mg_output}/data_quality/evidently_report_{name}.html")
def create_data_quality_df(obj_):
    df1=pd.DataFrame(obj_['data_quality']['data']['metrics']['reference']).T.reset_index().rename({'index':'feature'},axis=1)
    df2=pd.DataFrame(obj_['data_quality']['data']['metrics']['current']).T.reset_index().rename({'index':'feature'},axis=1)
    df1['set_data']='reference'
    df2['set_data']='current'
    rf= pd.concat([df1,df2])
    rf=rf.groupby(['feature','feature_type']).agg({'missing_count': lambda x: abs(list(x)[1]-list(x)[0]),
     'missing_percentage': lambda x: abs(list(x)[1]-list(x)[0]),
     'unique_count': lambda x: abs(list(x)[1]-list(x)[0]),
     'unique_percentage': lambda x: abs(list(x)[1]-list(x)[0]),
     # 'most_common_value': lambda x: 1 if list(x)[0]!=list(x)[1] else 0,
     # 'most_common_value_percentage': lambda x: 1 if list(x)[0]!=list(x)[1] else 0,
     'infinite_count': lambda x: abs(list(x)[1]-list(x)[0]),
     'infinite_percentage': lambda x: abs(list(x)[1]-list(x)[0]),
     'percentile_25': lambda x: abs(list(x)[1]-list(x)[0]),
     'percentile_50': lambda x: abs(list(x)[1]-list(x)[0]),
     'percentile_75': lambda x: abs(list(x)[1]-list(x)[0]),
     'max': lambda x: abs(list(x)[1]-list(x)[0]),
     'min': lambda x: abs(list(x)[1]-list(x)[0]),
     'mean': lambda x: abs(list(x)[1]-list(x)[0]),
     'std': lambda x: abs(list(x)[1]-list(x)[0]),
     'new_in_current_values_count': lambda x: abs(list(x)[1]-list(x)[0]),
     'unused_in_current_values_count': lambda x: abs(list(x)[1]-list(x)[0]),
    'new_in_current_values_count':'max', 'unused_in_current_values_count':'max',
                                               }
    ).reset_index()
    rf['unused_in_current_values_count'].fillna(0,inplace=True)
    rf['new_in_current_values_count'].fillna(0,inplace=True)
    indicators=['infinite_count',
       'percentile_50', 'max', 'mean',
       'missing_percentage', 'percentile_75', 'unused_in_current_values_count',
        'unique_percentage',
       'min', 'unique_count', 'percentile_25',
       'missing_count', 'new_in_current_values_count', 'std']
    rf=rf[['feature']+indicators]
    rf.rename(dict(zip(indicators,[f'difference_{x}' for x in indicators])),axis=1,inplace=True)
    return rf
def create_specific_drift(obj_):
    af=pd.DataFrame([{'feature':x,
                      'current_small_hist_0':obj_['data_drift']['data']['metrics'][x]['current_small_hist'][0],
     'current_small_hist_1':obj_['data_drift']['data']['metrics'][x]['current_small_hist'][1] ,
     'ref_small_hist_0':obj_['data_drift']['data']['metrics'][x]['ref_small_hist'][0],
     'ref_small_hist_1':obj_['data_drift']['data']['metrics'][x]['ref_small_hist'][1] ,
     # 'feature_type':obj_['data_drift']['data']['metrics'][x]['feature_type'],
      'p_value_drift':obj_['data_drift']['data']['metrics'][x]['p_value']
    } for x in obj_['data_drift']['data']['num_feature_names']+obj_['data_drift']['data']['cat_feature_names']])
    af['exist_drift']=af['p_value_drift'].apply(lambda x: 1 if x<(1. - obj_['data_drift']['data']['options']['confidence']) else 0)
    # af=af.merge(fi,how='left',on='feature') 
    for y in range(20):
        try:
            af[f'hist_ref_{y}']=af[f'ref_small_hist_0'].apply(lambda x: x[y])
            af[f'hist_current_{y}']=af[f'current_small_hist_0'].apply(lambda x: x[y])
        except: pass
    af.drop(['ref_small_hist_0','current_small_hist_0','current_small_hist_1','ref_small_hist_1'],axis=1,inplace=True)
    bf=create_data_quality_df(obj_)
    af=af.merge(bf,on='feature',how='left')    
    return af
def data_distribution_get_objects(df,cfg):  
    dfv=generate_color_feature_analysis(df,cfg)
    rf_list=[]    
    df[cfg['name_target']]=df[cfg['name_target']].map(dict(zip(range(len(cfg['sub_cols'])),cfg['sub_cols'])))
    for f in cfg['cat_features']+cfg['cont_features']:
        for y in ['name_target','name_prediction']:
            try:
                if f in cfg['cont_features']:
                    df[f]=pd.cut(df[f],10,labels=['P010','P020','P030','P040','P050','P060','P070','P080','P090','P100'])
                    _df,_=st_bars(df,f,cfg[y],cfg['sub_cols'])
                elif f in cfg['cat_features']:
                    if len(df[f].value_counts())>=10: dfx=df[df[f].isin(df[f].value_counts().index.tolist()[:10])]  
                    else: dfx=df.copy()
                    _df,_=st_bars(dfx,f,cfg[y],cfg['sub_cols'])
                _df['feature']=f
                del _df[f]
                _df['flg_target']=y
                _df['group_feature']=_df['group_feature'].astype(str)
                _df.rename({y:'name_classes',cfg['name_prediction']:'name_classes',cfg['name_target']:'name_classes','name_target':'name_classes','WG':'value'},axis=1,inplace=True)
                _df['metric']='WG'
                _df['type_metric']='report_general'
                # _df['name_classes']='ALL_VALUES'  
                rf_list.append(_df)
            except: pass
    try:return pd.concat([pd.concat(rf_list),pd.concat(dfv)],axis=0)
    except:return pd.concat(rf_list)