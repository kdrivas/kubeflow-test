import os
from .src.automl.utils import tryconvert
from .src.automl import model_outputs as Z
from .src.automl import pipeline_logger as P
from sklearn.preprocessing import LabelEncoder
from .src.aimg_exp import querys as Q
from .src.aimg_exp import mg_performance as S
from .src.aimg_exp import mg_data as D
from .src.aimg_exp import mg_interpretability as I
from .src.feedback import get_feedback
# from .src.aimg_exp.mg_performance import perf_fairness_metric,perf_features_metric,perf_do_plots,perf_hpo_iterations,perf_fpr_tpr_tresholds,perf_precision_recall_tresholds
# from .src.aimg_exp.mg_eda import plot_eda_distributions,plot_correlation
from datetime import datetime
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import log_loss,accuracy_score,roc_auc_score,confusion_matrix,classification_report, precision_score, recall_score, f1_score
from fairlearn.metrics import false_negative_rate, false_positive_rate,selection_rate
from .src.aimg_exp.mg_report import plot_correlation,plot_eda_distributions
# from aiutils import aiwr
class MG_Artifacts_Cls:
    """
    Register MG Artifacts in BigQuery & CloudStorage
    Ingesta de inkformación para el dashboard de monitoreo.
    Registra toda la información del modelo puesta en producción. 
    Almacena y supervisa los logs, metadata, metricas, modelos, artefactos, pipelines a lo largo del tiempo.    
    Monitor Metrics -> Performance, Valor, Data, Interpretabilidad, Costos, MetaInfo
    * Adaptable para la Evaluación del Modelo
    Args:
        df_train (pd.DataFrame): Pandas DataFrame.
        df_test (pd.DataFrame): Pandas DataFrame.
        model (pickle): Modelo Entrenado.
        preprocess (pickle): Preprocesamiento Usado.
        artifacts (pickle): Métricas, Objetos en el Entrenamiento.
        metadata (dictionary): Constantes.      
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
    Examples:
        mg=MG_Artifacts_Cls(df,df_test,model,preprocess,artifacts,metadata)
        mg.get_all_artifacts('kf_predict')
    Returns:
        Mg Artifacts
    """
    def __init__(self,df,df_test,model,preprocess,artifacts):
        self.model=model
        self.preprocess=preprocess
        self.artifacts=artifacts
        self.artifacts['metadata']['id_execution_mg']=datetime.now().__str__()[:19]
        self.artifacts['metadata']['n_classes']=df[self.artifacts['metadata']['name_target']].nunique()
        if self.artifacts['metadata']['n_classes']<=2:self.artifacts['metadata']['n_classes']=2
        self.dict_target= dict(zip(self.artifacts['metadata']['sub_cols'],range(self.artifacts['metadata']['n_classes'])))
        self.dict_target_inv= dict(zip(range(self.artifacts['metadata']['n_classes']),self.artifacts['metadata']['sub_cols']))       
        df[self.artifacts['metadata']['kpi_features']]=df[self.artifacts['metadata']['kpi_features']].astype(float)
        df_test[self.artifacts['metadata']['kpi_features']]=df_test[self.artifacts['metadata']['kpi_features']].astype(float)
        self.all_df=get_feedback(df,df_test,self.artifacts['metadata'],self.preprocess,self.model)
        self.x_train,self.y_train,self.id_train,self.x_test,self.y_test,self.id_test=self.split(self.all_df[self.all_df['set_analysis']=='training'],
                                                                                                self.all_df[self.all_df['set_analysis']=='predict'],
                                                                                                self.artifacts['metadata']['name_target'],
                                                                                                self.artifacts['metadata']['cat_features']+self.artifacts['metadata']['cont_features'],
                                                                                                self.artifacts['metadata']['id_features'],self.artifacts['metadata']['evol_features'])
        self.sv=shap.TreeExplainer(self.model)
        # self.svt=self.sv.shap_values(self.preprocess.transform(self.x_test[Z.cls_feature_name(self.model,self.artifacts['metadata']['type_model'])]))
        self.all_df=self.all_df[self.all_df['set_analysis']=='predict']
        if len(self.all_df[self.all_df[self.artifacts['metadata']['subpoblation_features']].isin(self.artifacts['metadata']['subpoblation_values'])])>5000:
            self.x_test_sample=self.all_df[self.all_df[self.artifacts['metadata']['subpoblation_features']].isin(self.artifacts['metadata']['subpoblation_values'])][self.artifacts['metadata']['name_features']].sample(5000)
            self.y_test_sample=self.all_df[self.all_df[self.artifacts['metadata']['subpoblation_features']].isin(self.artifacts['metadata']['subpoblation_values'])][self.artifacts['metadata']['name_target']].sample(5000)
            self.sub_cols_sample=self.all_df[self.all_df[self.artifacts['metadata']['subpoblation_features']].isin(self.artifacts['metadata']['subpoblation_values'])][self.artifacts['metadata']['sub_cols']].sample(5000)
        else: 
            self.x_test_sample=self.x_test
            self.y_test_sample=self.y_test
            self.sub_cols_sample=self.all_df[self.artifacts['metadata']['sub_cols']].copy()
        self.svv_filters=self.x_test_sample[self.artifacts['metadata']['subpoblation_features']].copy()
        self.svv=self.sv(pd.DataFrame(self.preprocess.transform(self.x_test_sample),columns=self.artifacts['metadata']['name_features'])[Z.cls_feature_name(self.model,self.artifacts['metadata']['type_model'])],check_additivity=False)
        # aiwr.save_dataset(self.all_df[self.all_df['set_analysis']=='predict'], f"bq://tmp.mg_poc_dataset_{self.artifacts['metadata']['id_modelo']}", if_exists='append')
    def split(self,df_train,df_test, target, features, index_features,datetime_feature):#
        x_tr,y_tr,id_tr=df_train[features+[self.artifacts['metadata']['evol_features']]+self.artifacts['metadata']['sub_cols']+[self.artifacts['metadata']['name_prediction']]].reset_index(drop=True),df_train[target].reset_index(drop=True), df_train[index_features].reset_index(drop=True)
        x_test,y_test,id_test=df_test[features+[self.artifacts['metadata']['evol_features']]+self.artifacts['metadata']['sub_cols']+[self.artifacts['metadata']['name_prediction']]].reset_index(drop=True),df_test[target].reset_index(drop=True), df_test[index_features].reset_index(drop=True)
        del df_train,df_test
        return x_tr,y_tr,id_tr,x_test,y_test,id_test
    def get_all_artifacts(self,source='kf_training'):
        self.artifacts['metadata']['source']=source
        mg= { 
            'perf__general_metrics':Q.bq_generate_and_save(S.exp_perf_general_metric(self.all_df.copy(),
                                                         self.artifacts['metadata']['name_target'],
                                                         self.artifacts['metadata']['name_prediction'],
                                                        self.artifacts['metadata']['sub_cols'],
                                                        self.artifacts['metadata']['subpoblation_features'],
                                                        self.artifacts['metadata']['kpi_features'],
                                                        self.artifacts['metadata']['supoblation_values']),
                                                           'mg_perf__general_metrics',self.artifacts['metadata']
                                                          ),
             'feature__explainer' :I.exp_feature_explainer(self.x_test_sample,
                                                     self.y_test_sample,self.sub_cols_sample,
                                                     self.svv,self.model,self.preprocess,
                                                     self.artifacts['metadata'],self.svv_filters,
                                                     self.artifacts['metadata']['supoblation_values']),
             'feature__drift' :D.exp_feature_drift(self.all_df,self.x_train,self.x_test,
                                                     self.y_train,self.y_test,
                                                     self.preprocess,
                                                     self.artifacts['metadata'])
            }
        if source=='kf_training': 
            obj_={}
            for u,v in self.artifacts['metadata'].items(): 
                if u not in ['hpo','group']: obj_[u]=str(v)      
            obj_['best_features']=mg['feature__explainer']['best_features']
            mg['model_info']= {'perf__training_validation':Q.bq_generate_and_save(pd.DataFrame(Z.cls_loss_training(self.model,                                                                                                                   self.artifacts['metadata']['metric'],self.artifacts['metadata']['type_model'])),
                                                             'mg_perf_metadata_training_validation',self.artifacts['metadata']
                ),
            'perf__hpo_trials':Q.bq_generate_and_save(tryconvert(self.artifacts['metadata']['hpo'],
                                                                 pd.DataFrame({'trial':[np.NAN],
                                                                               'value_trial':[np.NAN],
                                                                               'parameter':[np.NAN],
                                                                               'value':[np.NAN],
                                                                               'state_trial':[np.NAN]},index=[0]),
                                                                 lambda w: S.perf__hpo_trials(w)),'mg_perf_metadata_hpo_trials',self.artifacts['metadata'])} 
            mg['metadata']={'metadata__constants':Q.bq_generate_and_save(pd.DataFrame(obj_,index=[0]),
                                                                         'mg_metadata_constants',
                                                                         self.artifacts['metadata']),
            'metadata__costs':Q.bq_generate_and_save(self.artifacts['costs'][['cpu_usage_m_1','cpu_usage_0','cpu_usage_s_1','memory_usage_available','memory_usage_total','message','periodo','time_acum']],'mg_metadata_modelo_costos',self.artifacts['metadata'])
        }  
        mg['leaderboard']=Q.bq_generate_and_save(Q.create_leaderboard_dataframe(mg),'mg_leaderboard',self.artifacts['metadata'])
        return mg
    def generate_eda_reports(self): 
        x_test=self.all_df.copy() # [self.all_df['set_analysis']=='predict']
        mg_output=(self.artifacts['metadata']['id_modelo']+'/'+self.artifacts['metadata']['id_execution_mg']).replace(' ','').replace('-','').replace(':','')
        os.makedirs(f'./{mg_output}/drift', exist_ok = True)
        os.makedirs(f'./{mg_output}/target_distribution', exist_ok = True)
        os.makedirs(f'./{mg_output}/prediction_distribution', exist_ok = True)
        cols={'cat_features':self.artifacts['metadata']['cat_features'],'cont_features':self.artifacts['metadata']['cont_features']}
        plot_correlation(x_test[self.artifacts['metadata']['cat_features']+self.artifacts['metadata']['cont_features']],mg_output)        
        for load_var in self.artifacts['metadata']['cat_features']+self.artifacts['metadata']['cont_features']:
            try: plot_eda_distributions(x_test,cols,f'{mg_output}/drift',load_var,'set_analysis',name_fields=['train','test'],pal='Set1')
            except: print(f'Train/Test Analysis Error in {load_var}')
            try: plot_eda_distributions(x_test,cols,f'{mg_output}/target_distribution',load_var,self.artifacts['metadata']['name_target'],name_fields=self.artifacts['metadata']['sub_cols'],pal='mako')
            except: print(f'Real Target Analysis Error in {load_var}') 
            try: plot_eda_distributions(x_test,cols,f'{mg_output}/prediction_distribution',load_var,self.artifacts['metadata']['name_prediction'],name_fields=self.artifacts['metadata']['sub_cols'],pal='mako')
            except: print(f'Prediction Target Analysis Error in {load_var}') 

def register_mg_exp(df,df_test,dict_model,dict_preprocess,dict_artifacts,source='kf_predict'):
    if len(df)>=50000:df=df.sample(50000,random_state=60)
    if len(df_test)>=10000:df_test=df_test.sample(10000,random_state=60)
    df=df.loc[:,~df.columns.duplicated()]
    df_test=df_test.loc[:,~df_test.columns.duplicated()]
    df.reset_index(inplace=True,drop=True)
    df_test.reset_index(inplace=True,drop=True)
    return MG_Artifacts_Cls(df,df_test,dict_model,dict_preprocess,dict_artifacts).get_all_artifacts(source)
def register_mg_data(df,numerical_features,categorical_features,periodo,datetime_features='periodo',sub_cols=['general'],name_target='flg_general',name_prediction='flg_predict'):
    df['flg_general']=1
    df['flg_predict']=1
    for x in categorical_features:df[x]=LabelEncoder().fit_transform(df[x].fillna('nan'))
    x_train,x_test=df[df['periodo']==periodo][numerical_features+categorical_features],df[df['periodo']==periodo+pd.DateOffset(months=1)][numerical_features+categorical_features]
    y_train,y_test=df[df['periodo']==periodo][name_target],df[df['periodo']==periodo+pd.DateOffset(months=1)][name_target]
    _xf,xf,column_mapping=evidently_samples(x_train,y_train,
                                            x_test,y_test,sub_cols,
                                            name_target,name_prediction,numerical_features,categorical_features,datetime_features)
    # evidently_dashboard(_xf,xf,column_mapping,{id_execution},mg_output)
    obj_=evidently_profile(_xf,xf,column_mapping)    
    t=x_train.nunique()
    return {'cat_features':categorical_features,
            'cont_features':numerical_features,
            'bin_features':['flg_general','flg_predict']+list(x_train[list(t[t<=4].index)]),
        'evidently':evidently_get_objects(obj_,categorical_features,numerical_features,{}),
    'alibi_detect':alibi_detect_get_objects(_xf,xf,categorical_features,numerical_features,{})}