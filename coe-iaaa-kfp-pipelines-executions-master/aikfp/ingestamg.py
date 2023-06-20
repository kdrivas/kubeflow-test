from . import aiml, aimg,constants as C
from .src import executes as E
from .src import feedback as F
from aiutils import aiwr
import pandas as pd
def read_artifacts_and_datasets(df,id_project,_l_m):
    df.sort_values('createTime',inplace=True)   
    df=df[(df['name_subproject']==id_project)&(df['source']=='prediction')].tail(5)
    paths_models='gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/MODELS/'+C.mtd_pipeline[id_project]['name_prediction']+'/'+df['version'].iloc[0].replace('-','.')
    df_id=aiwr.read_dataset(C.mtd_pipeline[id_project]['query_feedback'])
    df_train=F.add_period_in_dataframe(df['paths_parquets_train'].iloc[0][0],str(df['period_of_training'].iloc[0])[:10],df_id,C.mtd_pipeline[id_project]['id_features'])
    # df_train=F.add_period_in_dataframe(df['paths_dataframes_train'].iloc[0][0],str(df['period_of_training'].iloc[0])[:10],df_id,C.mtd_pipeline[id_project]['id_features'],'aiwr') #df_train=aiwr.read_dataset(df['paths_dataframes_train'].iloc[0][0])
    df_pred=[sorted(x) for x in df['paths_parquets']]#df_prod=[pd.read_parquet([y for y in x if 'get-source' in y][0]) for x in df['paths_parquets']]
    df_pred=[F.add_period_in_dataframe([y for y in x if C.mtd_pipeline[id_project]['kf_name_of_dataset_to_score'] in y][0],str(df['createTime'].iloc[n])[:10],df_id,C.mtd_pipeline[id_project]['id_features'],'parquet') for n,x in enumerate(df_pred)]
    _l_m=[x for x in _l_m if df['version'].iloc[0].replace('-','.') in x]
    # models=[[y for y in _l_m if (('/'.join(w.split('/')[3:6])) in y) ] for w in paths_models]#df['paths_models'].iloc[0]
    # models=[x for x in models if len(x)!=0]
    # m=sum([list(set([m.split('/')[1] for m in model])) for model in models ],[])
    # _m={}
    models=[y for y in _l_m if (('/'.join(paths_models.split('/')[3:6])) in y)]#df['paths_models'].iloc[0]
    models=[x for x in models if len(x)!=0]
    _m={}
    for model in models:
        try:_m[model]=pd.read_pickle('gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/'+model)
        except Exception as e: pass  #preprocess no carga
    # for n in range(len(m)):
    #     _m[m[n]]=[pd.read_pickle('gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/'+model) for model in models[n]]
    return df_train,df_pred,_m
def save_metadata_mg_exp(id_project,

              df_pred,df_pred_postprocess,
              ):
    """
    df_train: data entrenamiento pd.DataFrame
    df_pred: data prediccion pd.DataFrame
    model: preprocess, model Object
    """
        # -- cargar data de entrenamiento y modelo (df,model)
        # --(ejecutar performance , real(), prediction(deliver))