from google.cloud import storage
from google.cloud.aiplatform import pipeline_jobs
from aiutils import aiwr
import pandas as pd
import numpy as np
import time
from collections import Counter
from .automl.utils import tryconvert
def get_list_kfpipelines(use_historical=True):
    bucket_name='ue4_ndlk_nonprod_stg_gcs_iadev_artfsto'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    _l=[x.name for x in blobs if ('PIPELINES/' in x.name) | ('MODELS/' in x.name)]
    _l_m=[x for x in _l if 'MODELS/' in x]
    _l=[x for x in _l if len(x.split('/'))==7]
    _l=[x for x in _l if ('prediction-' in x) | ('training-' in x)]
    # l=[x for x in l if '0-1-0' in x]
    _l=[x.split('/')[-2] for x in _l]
    if use_historical:
        m=aiwr.read_dataset("""bq:// SELECT id_pipeline FROM `rs-prd-dlk-sbx-fsia-8104.tmp.mg_list_kfpipelines`) WHERE EXECUTE_PREVIOUSLY <> 'YES' """)['id_pipeline'].to_list()
        _l=_l.difference(m)
    df=pd.DataFrame([{'id_pipeline':x,
                  'version':'-'.join(x.split('-')[-5:-2])
                  ,'name_pipeline':x.split('-')[-1],
                  'source':x.split('-')[-2],'name_subproject':'-'.join(x.split('-')[:-5])} for x in _l])
    # l=get_vtx_metadata(_l)
    l=pd.read_pickle('gs://ue4_ndlk_nonprod_stg_gcs_iadev_adsto/tmp/Ronald/projects/coe-iaaaa-iniciatives-model-governance/inputs/list_kf_ids')
    nf=pd.DataFrame({'id_pipeline':list(set(df['id_pipeline']).intersection(l.keys())), 
                     'state_pipeline':[l[x]['state'] for x in set(df['id_pipeline']).intersection(l.keys())],
                    'name_target':[tryconvert(x,np.NAN, lambda y: ['gs://ue4_ndlk_nonprod_stg_gcs_iadev_artfsto/MODELS/'+l[y]['pipelineSpec']['components']['comp-condition-register-outputs-1']['dag']['tasks']['register-model']['inputs']['parameters']['target']['runtimeValue']['constantValue']['stringValue']]) for x in set(df['id_pipeline']).intersection(l.keys())]
                    })
    df=df.merge(nf,how='left',on='id_pipeline')
    df['name_target']=df.apply(lambda y: tryconvert(y,np.NAN, lambda x: set([x['name_target'][0]+'/'+x['version'].replace('-','.')])),axis=1)
    nf=pd.DataFrame({'id_pipeline':[l[y]['displayName'] for y in l.keys()],
                 'createTime':[pd.to_datetime(l[y]['createTime']) for y in l.keys()],
                  'startTime':[pd.to_datetime(l[y]['startTime']) for y in l.keys()],
                  'endTime':[tryconvert(l, np.NAN,lambda l: pd.to_datetime(l[y]['endTime'])) for y in l.keys()],
                  'updateTime':[tryconvert(l, np.NAN,lambda l: pd.to_datetime(l[y]['updateTime'])) for y in l.keys()],
                 'n_nodes':[len(l[y]['jobDetail']['taskDetails']) for y in l.keys()],
                 'paths_parquets':[return_all_paths(l[y],'Dataset') for y in l.keys()],
                 'paths_artifacts':[return_all_paths(l[y],'Artifact') for y in l.keys()],
                     'paths_models':[set([return_models_path(x) for x in l[w]['jobDetail']['taskDetails'] if (str(return_models_path(x))!='nan') &('MODELS' in str(return_models_path(x))) ] ) for w in l.keys()]
                })
    df=df.merge(nf,how='left',on='id_pipeline')
    nf=df.groupby('name_subproject')[['name_target','paths_models']].agg(lambda x: [y for y in list(x) if (str(y)!='set()')&(str(y)!='nan')]).reset_index()
    del df['paths_models'],df['name_target']
    df=df.merge(nf,on='name_subproject')
    df['paths_models']=df['paths_models'].apply(lambda x: tryconvert(x,np.NAN,lambda y:set(y[0])))   
    df['name_target']=df['name_target'].apply(lambda x: tryconvert(x,np.NAN,lambda y:set(y[0])))   
    df['paths_dataframes_train']=df['paths_models'].apply(lambda x: tryconvert(x,np.NAN,lambda y:[f"bq://SELECT * FROM `rs-nprd-dlk-ia-dev-aif-d3d9.train.{w.split('/')[4]}`" for w in list(x)]))
    df['paths_dataframes_train'].fillna(df['name_target'].apply(lambda x: tryconvert(x,np.NAN,lambda y:[f"bq://SELECT * FROM `rs-nprd-dlk-ia-dev-aif-d3d9.train.{w.split('/')[4]}`" for w in list(x)])),inplace=True)
    # mf=df[(df['source']=='prediction')&(df['state_pipeline']=='PIPELINE_STATE_SUCCEEDED')].groupby(['name_subproject'])[['version']].agg([lambda x:Counter(x).most_common(1)[0][0],len]).reset_index()
    # mf=dict(zip(mf['name_subproject'],mf['version']['<lambda_0>']))    
    mf={'gestion-leads':'0-2-0',
        'gestion-leads-vehi':'0-2-0',
        'score-vehicular':'0-3-9',
'score-cliente-ami':'0-2-7',
'score-ingresos-nse':'0-2-1',
'upsell-vida':'0-1-0'}
    df=pd.concat([df[(df['name_subproject']==u)&(df['version']==v)&(df['state_pipeline']=='PIPELINE_STATE_SUCCEEDED')] for u,v in mf.items()])    
    nf=df[df['source']=='training'].groupby('name_subproject')['createTime'].agg('max').reset_index().rename({'createTime':'period_of_training'},axis=1)
    df=df.merge(nf,on='name_subproject',how='left')
    display(df.head(3))
    nf=df.sort_values('createTime').drop_duplicates(subset=['name_subproject'],keep='last')[['name_subproject','paths_parquets']].rename({'paths_parquets':'paths_parquets_train'},axis=1)
    df=df.merge(nf,on='name_subproject')
    df['period_of_training']=df['createTime'].apply(lambda x: str(x)[:10])
    df['EXECUTE_PREVIOUSLY']='NO'
    return df,_l_m
# l=[l[x] for x in l if l[x]['state']=='PIPELINE_STATE_SUCCEEDED']
def get_vtx_metadata(l):
    x={}
    start = time.time()
    for a in l:
        try:x[a]=pipeline_jobs.PipelineJob.get('projects/106504408617/locations/us-central1/pipelineJobs/'+a).to_dict()
        except:pass
        try:x[a]=pipeline_jobs.PipelineJob.get('projects/106504408617/locations/us-east4/pipelineJobs/'+a).to_dict()
        except: pass
    end= time.time()
    end-start
    return x

def return_all_paths(l,o_type):
    lst=[]
    name=l['displayName']
    _l=l['jobDetail']['taskDetails']
    for x in _l:
        try: 
            _x=x['outputs']
            for e in _x.keys():
                if (_x[e]['artifacts'][0]['schemaTitle']==f'system.{o_type}'):
                    lst.append(_x[e]['artifacts'][0]['uri'])
            if len(lst)==0:
                for e in _x.keys():
                    if (_x[e]['artifacts'][0]['schemaTitle']==f'system.{o_type}')&('df' in _x[e]['artifacts'][0]['uri']):
                        lst.append(_x[e]['artifacts'][0]['uri'])
        except: pass
    return lst
def return_models_path(x):
    try: return x['execution']['metadata']['output:Output']
    except: return np.NAN