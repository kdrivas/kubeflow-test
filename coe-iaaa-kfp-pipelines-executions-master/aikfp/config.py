import os
from .__metadata__ import (
    __URL__, __DESCRIPTION__, __BUSINESS__, __MLE__, __MLE_EMAIL__, __DS__, __DS_EMAIL__
)
__INFERENCE_TYPE__ = 'batch' # batch/online

################################ PACKAGE ################################
PACKAGE_NAME = 'aikfp'
PACKAGE_VERSION = '0.1.0'

def suffix_package_name():
    SUFFIX_PACKAGE = os.environ.get('DEBUG')
    if str(SUFFIX_PACKAGE).upper() == 'TRUE':
        return '-exp'
    else:
        return ''
PACKAGE_VERSION = '0.1.0' + suffix_package_name() # '' or '-exp'

PACKAGE_FILENAME = f'{PACKAGE_NAME}-{PACKAGE_VERSION}.tar.gz'

def get_package_uri(dir_path=False):
    GS_ARTIFACT_STORE = os.environ.get('GS_ARTIFACT_STORE')
    if dir_path:
        return '{}/PACKAGES/{}'.format(GS_ARTIFACT_STORE, PACKAGE_NAME)
    else:
        return '{}/PACKAGES/{}/{}'.format(GS_ARTIFACT_STORE, PACKAGE_NAME, PACKAGE_FILENAME)
        
def get_pipeline_uri():
    GS_ARTIFACT_STORE = os.environ.get('GS_ARTIFACT_STORE')
    return '{}/PIPELINES/{}/{}'.format(GS_ARTIFACT_STORE, PACKAGE_NAME, PACKAGE_VERSION)

def get_models_uri():
    GS_ARTIFACT_STORE = os.environ.get('GS_ARTIFACT_STORE')
    return '{}/MODELS'.format(GS_ARTIFACT_STORE)

def get_artifacts_uri():
    GS_ARTIFACT_STORE = os.environ.get('GS_ARTIFACT_STORE')
    return '{}/ARTIFACTS'.format(GS_ARTIFACT_STORE)

############################### ENVIORMENT ##############################
_envs = [
#     'ENV',
    'GOOGLE_CLOUD_PROJECT',
    'LOCATION',
#     'GS_ADS',
#     'GS_EDS',
    'GS_ARTIFACT_STORE',
#     'GS_LOG',
#     'SERVICE_ACCOUNT',
    'SECRET_GH_TOKEN',
#     'GH_TOKEN',
    'BQ_MODEL_GOVERNANCE',
    'BQ_FEATURE_STORE'
]

############################# SHARED PARAMS #############################
_features = [
    # Feature REF
    # Some Samples
    # "cli_pol:benef_ambulatorio",
    # "cli_pol:cie10_cap_cap21_servicios_salud",
    # "persona:ingreso_prom",
    # "persona:linea_tcmax",
  ]

############################### PIPELINES ###############################
train_pipeline = {
    "period": "", #"202105",
    "feature_ref": _features,
    "model_params": {
        # "task": "train",
        # "boosting_type": "gbdt",
        # "objective": "binary",
        # "metric": "auc",
        # "learning_rate": 0.05,
        # "feature_fraction": 0.9,
        # "bagging_fraction": 0.7,
        # "bagging_freq": 10,
        # "verbose": -1,
        # "max_depth": 4,
        # "num_leaves": 128,
        # "max_bin": 512
    },
    "register": False # True when automating
}

predict_pipeline = {
    "period": None,
    "feature_ref": _features,
    "extra_columns": [
        # "event_timestamp",
        # "upload_timestamp",
        # "cuc",
        # "id_poliza",
        # "model_version"
    ],
    "artifact_path": get_models_uri(),
    "register": True
}
