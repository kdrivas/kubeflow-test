# import fire
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component
from kfp.v2.dsl import Input, Output
from kfp.v2.dsl import Dataset, Artifact, Model, Metrics, ClassificationMetrics
from typing import NamedTuple
import os
import sys
from pathlib import Path
path = Path(os.path.abspath(__file__))
dirname = str(path.parent.absolute().parent.absolute())
sys.path.insert(1, dirname)
from aikfp import config
STEP = 'training'
BASE_IMAGE = "gcr.io/deeplearning-platform-release/base-cpu"
pipeline_name = '{}-{}-{}'.format(config.PACKAGE_NAME, config.PACKAGE_VERSION, STEP)\
                                .replace('_', '-').replace('.', '-')
pipeline_filename='{}_pipeline.json'.format(STEP)
pipeline_local_path = '{}/{}'.format('pipeline', pipeline_filename)
pipeline_path = '{}/{}'.format(config.get_pipeline_uri(), pipeline_filename)
pipeline_root = '{}/{}'.format(config.get_pipeline_uri(), 'pipeline_root')

enviorment_vars = {env: os.environ[env] for env in config._envs}

fixed_enviorment_vars = enviorment_vars.__str__()
fixed_package_uri = config.get_package_uri()
fixed_extra_code = """
import os
fixed_enviorment_vars = eval(fixed_enviorment_vars)
for env, val in fixed_enviorment_vars.items():
    os.environ[env] = val
os.system(f'''
    gcloud config set project $GOOGLE_CLOUD_PROJECT
    export GH_TOKEN=$(gcloud secrets versions access latest --secret="$SECRET_GH_TOKEN")
    gsutil cp "{fixed_package_uri}" package.tar.gz
    pip install package.tar.gz
    ''')
"""

@component(
    base_image=BASE_IMAGE
)
def load_inputs(
    fixed_enviorment_vars: str,
    fixed_package_uri: str,
    fixed_extra_code: str,
    df_pickle: Output[Artifact]
):
    exec(fixed_extra_code)
    from aikfp.ingestamg_simulation import read_datasets
    _all={}
    for _id in _ids:
        df, df_prod = read_datasets(_id)
        _all[_id]={}
        _all={_id:{'reference':df,'production':df_prod}}
    pd.to_pickle(_all,df_pickle.path)
@component(
    base_image=BASE_IMAGE
)
def register_exp(
    fixed_enviorment_vars: str,
    fixed_package_uri: str,
    fixed_extra_code: str,
    df_pickle: Input[Artifact],
):
    exec(fixed_extra_code)
    from aikfp.ingestamg_simulation import automl_simulation,aimg_simulation
    import pandas as pd
    df=pd.read_pickle(df_pickle.path)
    for _id in _ids:
        ml=automl_simulation(_id,df)
        for _df_prod in df_prod:
            mg= aimg_simulation(ml,_df_prod)
def compile_pipeline(package_path=pipeline_local_path):
    @dsl.pipeline(
        name=pipeline_name,
        description=config.__DESCRIPTION__,
        pipeline_root=pipeline_root,
    )
    def train_pipeline(
        fixed_enviorment_vars: str = fixed_enviorment_vars,
        fixed_package_uri: str = fixed_package_uri,
        fixed_extra_code: str = fixed_extra_code,
    ):
        load_inputs_op = load_inputs(
            fixed_enviorment_vars, fixed_package_uri, fixed_extra_code,
        )
        register_exp_op = register_exp(
            fixed_enviorment_vars, fixed_package_uri, fixed_extra_code,
            load_inputs_op.outputs['df_pickle']
        )
    compiler.Compiler().compile(
        pipeline_func=train_pipeline, package_path=package_path
    )
    print(package_path)
    # os.system(f'gsutil cp {pipeline_local_path} {pipeline_path}')
    # os.system(f'rm {pipeline_local_path}')
    # print(pipeline_path)

def run_pipeline(job_spec_path=pipeline_path, wait=True, enable_caching=False, **kargs):
    from kfp.v2.google.client import AIPlatformClient
    from aiutils.constants import c
    from aiutils.custom.vertex import wait_pipeline
    api_client = AIPlatformClient(
        project_id=c.PROJECT_ID,
        region=c.LOCATION,
    )
    
    labels = {
        'package_name': config.PACKAGE_NAME.lower(),
        'package_version': config.PACKAGE_VERSION.replace('.', '_').lower()
    }
    for meta in ['BUSINESS', 'MLE', 'DS', 'INFERENCE_TYPE']:
        if hasattr(config, '__{}__'.format(meta)):
            labels[meta.lower()] = getattr(config, '__{}__'.format(meta)).lower()

    response = api_client.create_run_from_job_spec(
        job_spec_path=job_spec_path,
        service_account=os.getenv('SERVICE_ACCOUNT'),
        enable_caching=enable_caching,
        labels=labels,
        **kargs
    )
    print(response['name'])
    if wait:
        state = wait_pipeline(response)
        print(state)
        assert state == 'PIPELINE_STATE_SUCCEEDED'
    return response['name']
    