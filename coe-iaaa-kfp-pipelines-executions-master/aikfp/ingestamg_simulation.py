from . import aiml, aimg,constants_simulation as C
import pandas as pd
def read_datasets(id_project):
    if id_project=='payment-projection-empresas':
        df=pd.read_parquet('gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/PIPELINES/payment_projection_empresas/0.1.0/pipeline_root/106504408617/payment-projection-empresas-0-1-0-training-20220308152023/feature-engineering_-7542317714031771648/df_transform_parquet')
        df.columns=[x.replace('#','n_') for x in df.columns]
        df.rename({'% de docs recuperados - real':'target'},axis=1,inplace=True)
        df_prod=df[df['periodo']>'2021-09-01']
        df=df[(df['periodo']<='2021-09-01')&(df['periodo']>='2021-05-01')]
    elif id_project=='churn-empresas':
        df=pd.read_parquet('gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/PIPELINES/churn-empresas/0.1.0/pipeline_root/106504408617/churn-empresas-0-1-0-training-20220326042600/feature-engineering_-536683720145371136/df_transform_parquet')
        df.rename({'id_poliza_':'id_poliza'},axis=1,inplace=True)
        df_prod=df[df['periodo']>'2021-05-01']
        df=df[df['periodo']<='2021-05-01']
    elif id_project=='laft-empresas':
        df=pd.read_parquet('gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/PIPELINES/laft_empresas/0.3.5/pipeline_root/106504408617/laft-empresas-0-3-5-training-20220430065812/prepare-dataset_-3851084556262178816/df_parquet')
        df_prod=df[df['periodo']>'2021-12-01']
        df=df[df['periodo']<='2021-12-01']
    elif id_project=='gestion-leads':
        df=pd.read_parquet('gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/PIPELINES/gestion_leads_vehi/0.1.0/pipeline_root/106504408617/gestion-leads-vehi-0-1-0-training-20220211025413/target-engineering_-8667676661153529856/df_parquet')
        # df=pd.concat([df[df['cuenta_ganado_lead']==0].sample(frac=0.5),df[df['cuenta_ganado_lead']==1]])
        df['periodo']=df['fechalead'].apply(lambda x: pd.to_datetime(str(x)[:8]+'01'))
        df_prod=df[df['periodo']>'2021-06-01']
        df=df[df['periodo']<='2021-06-01']
    elif id_project=='score-vehicular-renov-persistencia':
        df=pd.read_parquet('gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/PIPELINES/score_vehicular_renov/0.3.7/pipeline_root/106504408617/score-vehicular-renov-0-3-7-training-20220526191231/enrich_1212242905704431616/df_enrich_parquet')
        df.rename({'perveh_persistenciarenov_vehicular_cls':'target'},axis=1,inplace=True)
        df['des_origen']=df['des_origen'].apply(lambda x: str(x).split(' ')[0])
        df_prod=df[(df['periodo']>'2021-06-01')&(df['periodo']<'2022-03-01')]
        df=df[df['periodo']=='2021-06-01']    
    elif id_project=='churn-empresas':
        df=pd.read_parquet('data_entries/churn-empresas')
        df.rename({'id_poliza_':'id_poliza'},axis=1,inplace=True)
        df_prod=df[df['periodo']>'2021-06-01']
        df=df[df['periodo']<='2021-06-01']
    if len(df)>500000: df=df.sample(500000,random_state=60).copy()
    if len(df_prod)>500000: df_prod=df_prod.sample(500000,random_state=60).copy()
    df_prod.sort_values('periodo',inplace=True)
    list_periods=df_prod['periodo'].unique()
    df_prod=[df_prod[df_prod['periodo']==x] for x in list_periods]
    return df,df_prod
def automl_training(id_project,df):
    # Read Dataset and Artifacts
    # df,df_prod=read_datasets(id_project)
    # Save Project
    # aiwr.save_dataset(pd.DataFrame(C.mtd_project[id_project],index=[0]), f"bq://tmp.mg_poc_project", if_exists='append')
    # Training
    ml=aiml.MlOptimization_cls(df, C.mtd_pipeline[id_project])
    if C.mtd_pipeline[id_project]['steps']['use_preprocess']: ml.preprocess_dataset(C.mtd_pipeline[id_project]['preprocess'])
    if C.mtd_pipeline[id_project]['steps']['use_fs']: ml.feature_selection(C.mtd_pipeline[id_project]['feature_selection'])
    if C.mtd_pipeline[id_project]['steps']['use_hpo']: ml.tuning(C.mtd_pipeline[id_project]['tuning'])
    if C.mtd_pipeline[id_project]['steps']['use_training']: ml.training(C.mtd_pipeline[id_project]['training'])
    ml.save_objects_and_finalize()
    return ml
def save_metadata_mg_exp(df_prod,ml):
    mg=[]
    # Evaluate
    mg.append(ml.evaluate())
    # Monitoring in production data
    for x in df_prod:
        mg.append(ml.monitor(x))    
    return mg