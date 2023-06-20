import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from datetime import datetime
from sklearn.metrics import log_loss,accuracy_score,roc_auc_score
import optuna
import os
from .src.automl import model_outputs as Z
from .src.automl import training_all as A
from .src.automl import model_selection as M
from .src.automl import pipeline_logger as P
from .src.automl import hpo_tuning as H
from .src.automl import pipeline_cfg as C
from .src.automl import transformers as T
from .src.automl import utils as U
from .src.feedback import get_feedback
from . import aimg
# from aiutils import aiwr
import hashlib
class MlOptimization_cls:
    """
    AutoMl End-To-End
    Baseline de los Proyectos - SubProyectos de la Coe AI. 
    Registra toda la información de la propuesta entrenada. 
    Almacena y supervisa los logs, metadata, metricas, modelos, artefactos, pipelines de un experimento.    
    Preprocesamiento -> Feature Selection -> HPO -> Model Training -> Evaluate -> Monitoring
    Args:
        df (pd.DataFrame): Pandas DataFrame.
        cfg (dictionary): Configuración del experimento.
         (obligatory)
                 name_project: Nombre del Proyecto/ Link de repositorio de Github
                 name_subproject: Nombre del SubProyecto
                 task: Tarea por realizar (Ej. Classification, Regression, Clustering, One-Shot)
                 version: Version del modelo en .config
                 name_target: Nombre del Target en la data consolidada
                 name_prediction: Nombre del Target Solucion (Ej. per_payprojection_empresas_cls)
                 id_features (list): Nombre de los Indices de la data consolidada (Ej id_contratante, ramo, id_poliza)
                 kpi_features (list): Nombre de los KPI Features, representan una ponderación con el Target (Ej. Monto_Dolares, Status -> Monto Recuperado)
                 subpoblation_features (list): Manera de filtar las metricas, representa los grupos de rendimiento por visualizar (Ej. Tipo_Canal, NSE)
                 type_model: Modelo elegido (Ej. LGB, XGB, CTB)
                 cfg_parameters (dictionary): Parametros Sugeridos
                 selection_method: Separación train/test
                 obs_features (list): Nombre de los Indices, a nivel de observación, de la data consolidada (Ej id_contratante)
                 cat_features (list): Variables Categóricas
                 cont_features (list): Variables Numéricas
                 text_features (list): Variables de Texto
    Methods:
        preprocess
         steps_cat_preprocess (list): Transformers por usar en Variables Categóricas
         steps_cont_preprocess (list): Transformers por usar en Variables Continuas
         preprocess_constants (dictionary): Constants in Transformers
        feature_selection
         fs_treshold: Porcentaje de variables por mantener
        tuning
         n_trials: Número de trials usados en el HPO
        training
         n_folds: Número de folds usados
    Examples:
            ml=aiml.MlOptimization_cls(df,cfg)
            # ml.feature_engineering()
            ml.preprocess_dataset()
            ml.feature_selection(0.9)
            ml.tuning(20)
            ml.training(5)
            ml.evaluate()
            ml.register_mg_exp(df_test)
    Returns:
        AutoMl
    """
    def __init__(self, df,cfg):
        self.cfg=cfg.copy()
        U.exist_element_cfg(df,'periodo',datetime.now().__str__()[:8]+'01')
        df['periodo']=pd.to_datetime(df['periodo'])
        df.sort_values('periodo',inplace=True)
        df.reset_index(inplace=True,drop=True)
        df.columns=[x.replace('#','_').replace('%','_').replace('[','_').replace(']','_').replace(' ','_') for x in df.columns]
        self.df=df
        if self.cfg['framework_model']=='NLP':
            self.x_train,self.y_train=self.df[self.cfg['text_features']],self.df[self.cfg['name_target']]#
            self.x_tr, self.x_vl, self.y_tr, self.y_vl=M.selection_method(self.x_train, self.y_train,self.cfg) #
        self.cfg,self.logger=C.generate_metadata(self.df,self.cfg)
        self.workflow=f"gs://ue4_ndlk_nonprod_stg_gcs_iadev_adsto/tmp/Ronald/projects/coe-iaaaa-iniciatives-model-governance/ml_artifacts/{self.cfg['name_project']}/{self.cfg['id_entrenamiento']}"# consider subproject
    def preprocess_dataset(self,_cfg):
        self.cfg['cat_features']=list(set([ x.split('_ohe_')[0] for x in self.cfg['cat_features']]))
        self.cfg['n_used_features']=len(self.cfg['cat_features'])+len(self.cfg['cont_features'])
        self.x_train,self.y_train=self.df[self.cfg['cat_features']+self.cfg['cont_features']],self.df[self.cfg['name_target']]
        self.cfg['steps_cat_preprocess']=_cfg['steps_cat_preprocess']
        self.cfg['steps_cont_preprocess']=_cfg['steps_cont_preprocess']
        self.cfg['preprocess_constants']=_cfg['preprocess_constants']
        self.x_train,self.preprocess=T.preprocess(self.x_train,self.cfg,self.logger)
        self.x_tr, self.x_vl, self.y_tr, self.y_vl=M.selection_method(self.x_train, self.y_train,self.cfg)  
        if self.cfg['fix_selection_method']=='si':self.x_tr,self.y_tr,self.x_vl,self.y_vl=M.fix_selection_method(self.x_tr,self.y_tr,self.x_vl,self.y_vl,self.df,self.cfg['id_features'])
    def feature_selection(self,_cfg):
        self.cfg['use_fs']='si'
        self.cfg['fs_mode']=_cfg['fs_mode']
        self.cfg['fs_treshold']=_cfg['fs_treshold']
        self.training({'n_folds':5})            
        nf=pd.concat([Z.cls_feature_importance(self._obj[x]['model'],self.cfg['type_model']) for x in range(len(self._obj))]).groupby('feature').agg('mean').reset_index()
        if self.cfg['fs_mode']=='gain': nf=nf[nf['importance']>=self.cfg['fs_treshold']]['feature']
        elif self.cfg['fs_mode']=='n_features': 
            n=int(self.cfg['fs_treshold']*self.cfg['n_features'])
            nf=nf.sort_values('importance',ascending=False).iloc[:n]['feature']
        self.cfg['cat_features']=list(set(self.cfg['cat_features']).intersection(nf))
        self.cfg['cont_features']=list(set(self.cfg['cont_features']).intersection(nf))
        P.save_system_line(self.logger,f"FS done, new_cat_features: {self.cfg['cat_features']}, new_cont_features: {self.cfg['cont_features']}")
        self.preprocess_dataset({'steps_cat_preprocess':self.cfg['steps_cat_preprocess'],
                                 'steps_cont_preprocess':self.cfg['steps_cont_preprocess'],
                                 'preprocess_constants':self.cfg['preprocess_constants']})
    def tuning(self,_cfg):
        self.cfg['use_hpo']='si'
        self.cfg['n_trials']=_cfg['n_trials']
        self.cfg['framework_hpo']='optuna'
        self.study = optuna.create_study(direction="minimize", study_name=f"{self.cfg['type_model']} Classifier")
        try: self.study.enqueue_trial(self.cfg['hpo'].best_params)
        except: pass
        self.study.optimize(lambda x: H.hpo_objective(x,self.x_tr,self.y_tr,self.x_vl,self.y_vl,self.cfg,self.logger), n_trials=self.cfg['n_trials'])  
        self.cfg['list_of_parameters']=[x for x in self.study.trials_dataframe().columns if 'params_' in x]
        self.cfg['hpo']=self.study
    def training(self,_cfg):
        self.cfg['n_folds']=_cfg['n_folds']
        if self.cfg['n_folds']<=2:
            self.cfg['strategy_fold']='no_folds'
            self.skf= [(np.array(self.x_tr.index.tolist()),np.array(self.x_vl.index.tolist()))]
        else:
            self.cfg['strategy_fold']='use_folds'
            self.skf=M.kfold_selection_method(self.x_train,self.y_train,self.cfg)
        self._obj=[]
        for fold,(idx_tr,idx_vl) in enumerate(list(self.skf)):
            P.save_system_line(self.logger,f'Resultados en fold {fold+1}')
            x_tr, x_vl=self.x_train.iloc[idx_tr],self.x_train.iloc[idx_vl]
            y_tr, y_vl=self.y_train.iloc[idx_tr],self.y_train.iloc[idx_vl] 
            if self.cfg['fix_selection_method']=='si':x_tr,y_tr,x_vl,y_vl=M.fix_selection_method(x_tr,y_tr,x_vl,y_vl,self.df,self.cfg['id_features'])
            try:obj=A.save_artifacts_and_logs(x_tr,y_tr,x_vl,y_vl,self.study.best_params,self.cfg,self.logger)
            except:obj=A.save_artifacts_and_logs(x_tr,y_tr,x_vl,y_vl,{'random_state':self.cfg['cfg_parameters']['seed']},self.cfg,self.logger)
            self._obj.append(obj)
    def save_objects_and_finalize(self):     
        self.dict_model=self._obj[0]['model']
        self.artifacts={}
        self.artifacts['costs']= P.calculate_costs('logger')  # Pendiente: añadir costo de todos los nodos          
        self.cfg['cat_features']=list(set([ x.split('_ohe_')[0] for x in self.cfg['cat_features']])) # Pendiente: Optimizar OHE
        self.artifacts['metadata']= self.cfg                
        # self.df.to_parquet(f"{self.workflow}/dataset_train.csv",index=False)
        # aiwr.save_dataset(df, f"bq://tmp.mg_poc_dataset_{self.workflow}", if_exists='append')
        pd.to_pickle(self.dict_model,f"{self.workflow}/model.pkl")
        pd.to_pickle(self.preprocess,f"{self.workflow}/preprocess.pkl")
        pd.to_pickle(self.artifacts,
                     f"{self.workflow}/artifacts.pkl")
        # ml.logger=np.NAN
        # ml.skf=np.NAN
        # pd.to_pickle(self,f"{self.workflow}/pipeline.pkl")
    def evaluate(self):
        mg=aimg.register_mg_exp(self.df.iloc[self.x_tr.index].reset_index(drop=True),
                            self.df.iloc[self.x_vl.index].reset_index(drop=True),
                                self.dict_model,self.preprocess,self.artifacts,
                                'kf_training' )  
        P.save_system_line(self.logger,f'Evaluate model')
        return mg    
    def monitor(self,df_test):
        x=str(df_test['periodo'].iloc[0])[:10]
        mg=aimg.register_mg_exp(self.df[self.df['periodo']==self.df['periodo'].max()].copy(),
                                df_test,
                                self.dict_model,self.preprocess,self.artifacts,
                                'kf_predict')
        # P.save_system_line(self.logger,f'MG Ejecucion {x}')
        # pd.to_pickle(mg,f"{self.cfg['workflow']}/mg_{self.cfg['name_project']}_{self.cfg['name_subproject']}_{x}.pkl")
        return mg