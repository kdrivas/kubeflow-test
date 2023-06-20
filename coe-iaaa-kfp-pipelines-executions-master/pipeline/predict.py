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

from my_package import config

STEP = 'prediction'
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
def get_source(
    fixed_enviorment_vars: str,
    fixed_package_uri: str,
    fixed_extra_code: str,
    period: str,
    df_parquet: Output[Dataset]
) -> NamedTuple(
    'Outputs',[
        ('df_bq', str)]
):
    exec(fixed_extra_code)
    ##################################################################################
    from my_package.predict import get_source
    df, df_bq = get_source(period)
    ##################################################################################
    df.to_parquet(df_parquet.path)
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['df_bq'])
    return outputs(df_bq)

@component(
    base_image=BASE_IMAGE
)
def enrich(
    fixed_enviorment_vars: str,
    fixed_package_uri: str,
    fixed_extra_code: str,
    df_bq: str,
    feature_ref: str,
    df_enrich_parquet: Output[Dataset]
) -> NamedTuple(
    'Outputs',[
        ('df_enrich_bq', str)]
):
    exec(fixed_extra_code)
    feature_ref = eval(feature_ref)
    ##################################################################################
    from my_package.predict import enrich    
    df_enrich, df_enrich_bq = enrich(
        entity_source=df_bq,
        feature_ref=feature_ref
    )
    ##################################################################################
    df_enrich.to_parquet(df_enrich_parquet.path)
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['df_enrich_bq'])
    return outputs(df_enrich_bq)

@component(
    base_image=BASE_IMAGE
)
def preprocess_and_predict(
    fixed_enviorment_vars: str,
    fixed_package_uri: str,
    fixed_extra_code: str,
    df_parquet: Input[Dataset],
    artifact_path: str,
    # targets: str,   # For Multiple Targets
    df_predicted_parquet: Output[Dataset],
    df_report_html: Output[Artifact],
    df_transform_report_html: Output[Artifact]
):
    exec(fixed_extra_code)
    # targets = eval(targets)
    import pandas as pd
    df = pd.read_parquet(df_parquet.path)
    ##################################################################################
    from my_package.predict import preprocess_and_predict    
    df_predicted = preprocess_and_predict(
        df,
        artifact_path,
        # targets,
        df_report_html.path,
        df_transform_report_html.path
    )
    ##################################################################################
    df_predicted.to_parquet(df_predicted_parquet.path)

@component(
    base_image=BASE_IMAGE
)
def save_predictions(
    fixed_enviorment_vars: str,
    fixed_package_uri: str,
    fixed_extra_code: str,
    df_parquet: Input[Dataset],
    period: str,
    extra_columns: str,
    # targets: str    # For Multiple Targets
) -> str:
    exec(fixed_extra_code)
    extra_columns = eval(extra_columns)
    # targets = eval(targets)
    import pandas as pd
    df = pd.read_parquet(df_parquet.path)
    ##################################################################################
    from my_package.predict import save_predictions    
    table_saved = save_predictions(
        df,
        period,
        extra_columns,
        # targets
    )
    ##################################################################################
    return table_saved

def compile_pipeline(package_path=pipeline_local_path):
    @dsl.pipeline(
        name=pipeline_name,
        description=config.__DESCRIPTION__,
        pipeline_root=pipeline_root,
    )
    def predict_pipeline(
        fixed_enviorment_vars: str = fixed_enviorment_vars,
        fixed_package_uri: str = fixed_package_uri,
        fixed_extra_code: str = fixed_extra_code,
        period: str = config.predict_pipeline['period'].__str__(),
        feature_ref: str = config.predict_pipeline['feature_ref'].__str__(),
        artifact_path: str = config.predict_pipeline['artifact_path'].__str__(),
        extra_columns: str = config.predict_pipeline['extra_columns'].__str__(),
        register: str = config.predict_pipeline['register'].__str__()
    ):
        get_source_op = get_source(
            fixed_enviorment_vars, fixed_package_uri, fixed_extra_code,
            period)
        enrich_op = enrich(
            fixed_enviorment_vars, fixed_package_uri, fixed_extra_code,
            get_source_op.outputs['df_bq'],
            feature_ref
        )
        # TARGETS = [
        #     
        # ]   # For Multiple Targets
        preprocess_and_predict_op = preprocess_and_predict(
            fixed_enviorment_vars, fixed_package_uri, fixed_extra_code,
            enrich_op.outputs['df_enrich_parquet'],
            artifact_path,
            # TARGETS.__str__()
        )
        save_predictions_op = save_predictions(
            fixed_enviorment_vars, fixed_package_uri, fixed_extra_code,
            preprocess_and_predict_op.outputs['df_predicted_parquet'],
            period,
            extra_columns,
            # TARGETS.__str__()
        )        

    compiler.Compiler().compile(
        pipeline_func=predict_pipeline, package_path=package_path
    )
    print(package_path)

def run_pipeline(job_spec_path=pipeline_path, wait=True, enable_caching=False, **kargs):
    from kfp.v2.google.client import AIPlatformClient
    from aiutils.constants import c
    from aiutils.custom.vertex import wait_pipeline
    api_client = AIPlatformClient(
        project_id=c.PROJECT_ID,
        region=c.LOCATION,
    )
#     response = api_client.create_schedule_from_job_spec(
#         job_spec_path=pipeline_path,
#         schedule='0 2 * * 6', # At 02:00 AM, only on Saturday
#         time_zone='America/Lima',
#     )
#     print(response['name'])
    response = api_client.create_run_from_job_spec(
        job_spec_path=job_spec_path,
        service_account=os.getenv('SERVICE_ACCOUNT'),
        enable_caching=enable_caching,
        **kargs
    )
    print(response['name'])
    if wait:
        state = wait_pipeline(response)
        print(state)
        assert state == 'PIPELINE_STATE_SUCCEEDED'
    return response['name']
    