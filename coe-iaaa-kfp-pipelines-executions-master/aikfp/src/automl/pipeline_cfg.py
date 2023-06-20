from . import utils as U
from . import pipeline_logger as P
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
def generate_metadata(df,cfg, period_not_exist=True,finalize_event=False):
        workflow='gs://ue4_ndlk_nonprod_stg_gcs_iadev_adsto/tmp/Ronald/projects/coe-iaaaa-iniciatives-model-governance'
        if period_not_exist: 
            cfg['id_entrenamiento']=datetime.now().__str__()[:19]
        hash_m = hashlib.md5()
        hash_m.update(f"{cfg['name_project']}_{cfg['name_subproject']}_{cfg['type_model']}_{cfg['version']}_{cfg['id_entrenamiento']}".encode("utf8"))       
        cfg['id_modelo']=str(int(hash_m.hexdigest(), 16))[0:15]
        cfg['workflow']=f"{workflow}/{cfg['name_project']}/{cfg['name_subproject']}/{cfg['id_modelo']}"
        # df=clean_df(df)
        logger=P.save_system_logs('logger')
        cfg['n_classes']=df[cfg['name_target']].nunique()
        cfg['periodo_min']=df['periodo'].min().__str__()[:10]
        cfg['periodo_max']=df['periodo'].max().__str__()[:10]
        cfg['n_rows']=len(df)
        U.exist_element_cfg(cfg,'cat_features',list(set(df.select_dtypes(include=object)).difference([cfg['name_target']]+cfg['id_features'])))
        U.exist_element_cfg(cfg,'cont_features',list(set(df.select_dtypes(exclude=object)).difference([cfg['name_target']]+cfg['id_features'])))
        U.exist_element_cfg(cfg,'evol_features','periodo')
        U.exist_element_cfg(cfg,'text_features',[])
        U.exist_element_cfg(cfg,'obs_features',cfg['id_features'])
        U.exist_element_cfg(cfg,'sub_cols',[f"{cfg['name_prediction']}_prob_{x}" for x in range(df[cfg['name_target']].nunique())])
        U.exist_element_cfg(cfg,'type_model','LGB')
        U.exist_element_cfg(cfg,'selection_method','stratified')
        U.exist_element_cfg(cfg,'cfg_parameters',{})
        U.exist_element_cfg(cfg['cfg_parameters'],'l_rate',0.06)
        U.exist_element_cfg(cfg['cfg_parameters'],'n_estimators',6000)
        U.exist_element_cfg(cfg['cfg_parameters'],'early_stopping_rounds',5000)
        U.exist_element_cfg(cfg['cfg_parameters'],'seed',60)
        U.exist_element_cfg(cfg['cfg_parameters'],'test_size',0.33)
        U.exist_element_cfg(cfg['cfg_parameters'],'n_estimators',3000)
        U.exist_element_cfg(cfg,'list_of_parameters',list(cfg['cfg_parameters'].keys()))
        U.exist_element_cfg(cfg,'fix_selection_method','no')
        U.exist_element_cfg(cfg,'use_fs','no')
        U.exist_element_cfg(cfg,'fs_treshold',1)
        U.exist_element_cfg(cfg,'fs_mode','n_features')
        U.exist_element_cfg(cfg,'use_hpo','no')
        U.exist_element_cfg(cfg,'n_trials',0)
        U.exist_element_cfg(cfg,'hpo',np.NAN)
        U.exist_element_cfg(cfg,'supoblation_values',list(df[cfg['subpoblation_features']].value_counts().index[:3]))
        cfg['name_features']=cfg['cat_features']+cfg['cont_features']
        cfg['n_features']=len(cfg['cat_features'])+len(cfg['cont_features'])
        cfg['n_ids']=len(df[cfg['id_features']].value_counts())
        cfg['n_obs']=len(df[cfg['obs_features']].value_counts())
        try:cfg['group']=df['periodo'].dt.month
        except:cfg['group']=np.NAN
        if cfg['type_model']=='LGB':
            if cfg['n_classes']<=2:
                cfg['n_classes']=1
                cfg['obj']='binary'
                cfg['metric']='binary_logloss'    
            else:
                cfg['obj']='multiclass'
                cfg['metric']='multi_logloss'
        elif cfg['type_model']=='XGB':
            cfg['obj']='multi:softprob'
            cfg['metric']='mlogloss'
        elif cfg['type_model']=='CTB':
            cfg['obj']='MultiClass'
            cfg['metric']='MultiClass'
        P.save_system_line(logger,f'Cfg de entrenamiento creado')
        if finalize_event:
            _cfg={}
            _cfg['metadata']=cfg
            _cfg['costs']=P.calculate_costs('logger')
        else: _cfg=cfg.copy()
        return _cfg,logger