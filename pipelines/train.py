import kfp 
from kfp import dsl, compiler, Client
from kfp.dsl import Input, Output, Dataset, Metrics, Model
# from kfp.dsl import 

# @dsl.component
# def validate():
#     pass

PIPELINE_NAME = "pipeline-xgboost-test"
PIPELINE_PACKAGE_PATH = "training_pipeline.yaml"
PIPELINE_VERSION_NAME_BASE = "pipeline-xgboost-test"

@dsl.component(base_image="index.docker.io/seyco/base_kbf_image")
def preprocess(
    raw_file: str,
    prep_file: Output[Dataset],
) -> None:
    """
    Preprocess raw file

    Args:
        raw_file: the file that gathered all the raw data
        prep_file: the output file that contains basic preprocessing
        steps

    Return:
        None
    """
    import pandas as pd

    df_input = pd.DataFrame({'points': [0, 1, 0, 0, 0, 1, 1, 1, 0, 0], 'country_city': ['peru,comas', 'peru,lince', 'peru,los olivos', 'peru,san juan de lurigancho', 'peru,comas', 'peru,lince', 'peru,los olivos', 'peru,san juan de lurigancho', 'peru,comas', 'peru,lince'], 'cap_1': [0.639, 0.109, 0.63, 0.026, 0.53, 0.889, 0.874, 0.196, 0.891, 0.915], 'cap_2': [0.466, 0.394, 0.12, 0.571, 0.225, 0.681, 0.459, 0.832, 0.019, 0.975], 'cap_3': [0.867, 0.965, 0.961, 0.249, 0.993, 0.218, 0.4, 0.916, 0.516, 0.855], 'cap_4': [0.983, 0.658, 0.351, 0.551, 0.451, 0.007, 0.6, 0.055, 0.73, 0.017], 'cap_5': [0.689, 0.333, 0.63, 0.772, 0.389, 0.885, 0.311, 0.03, 0.253, 0.164], 'cap_6': [0.673, 0.079, 0.987, 0.404, 0.357, 0.55, 0.981, 0.641, 0.042, 0.894]})
    df_input[["country", "city"]] = df_input["country_city"].apply(lambda x: pd.Series(x.split(",")))
    df_input.to_csv(prep_file.path, index=False)


@dsl.component(base_image="index.docker.io/seyco/base_kbf_image")
def engineer_features(
    prep_file: Input[Dataset],
    feat_file: Output[Dataset],
) -> None:
    """
    Create dummy features

    Args:
        prep_file: the preprocessed file

    Return:
        feat_train_file: the features splitted by train
        feat_test_file: the features splitted by test
        feat_valid_file: the features splitted by valid
    """
    import pandas as pd

    df_prep = pd.read_csv(prep_file.path)

    # Create dummy features
    df_prep["cap_1_cap_2"] = df_prep["cap_1"] * df_prep["cap_2"]
    df_prep["cap_4_cap_2"] = df_prep["cap_4"] * df_prep["cap_2"]
    df_prep["cap_3_cap_1"] = df_prep["cap_3"] * df_prep["cap_1"]

    # Removing string columns
    df_prep = df_prep.drop(columns=["country_city", "country", "city"])

    # Rename columns
    df_prep = df_prep.rename(columns={"points": "target"})

    df_prep.to_csv(feat_file.path, index=False)


@dsl.component(base_image="index.docker.io/seyco/base_kbf_image")
def train(
    feat_file: Input[Dataset],
    metrics: Output[Metrics],
    model: Output[Model],
    train_size: float = 0.80,
):
    """
    Train an XGBoost classifier

    Args:
        train_file: the dataset that contains the train data
        test_file: the dataset that contains the test data
        valid_file: the dataset that contains the valid data
    Return:

    """
    import xgboost
    import pickle
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    if train_size < 0 and train_size > 0.8:
        raise Exception("Invalid train size... should be less than 0.8 and greater than 0") 

    df_data = pd.read_csv(feat_file.path)

    x_rest, x_test, y_rest, y_test = train_test_split(df_data.drop(columns="target"), df_data[["target"]], test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_rest, y_rest, test_size=0.1)

    params = {
        "n_estimators": 2, 
        "max_depth": 3, 
        "learning_rate": 1, 
        "objective": "binary:logistic",
    }
    bst = xgboost.XGBClassifier(**params, random_state=0)
    # fit model
    bst.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    # make predictions
    preds = bst.predict_proba(x_test)[:, 1]

    metrics.log_metric("accuracy",roc_auc_score(y_test, preds))
    metrics.log_metric("model", "XGBoost")
    metrics.log_metric("dataset_size", len(x_train))

    with open(model.path, "wb") as model_path:
        pickle.dump(bst, model_path)


# @dsl.component
# def evaluate():
#     pass


@dsl.pipeline(name=PIPELINE_NAME, description="An example pipeline.")
def training_pipeline() -> None:
    prep = preprocess(raw_file="people.csv")
    eng_feat = engineer_features(
        prep_file=prep.outputs["prep_file"]
    )
    train(
        feat_file=eng_feat.outputs["feat_file"],
    )


def build_pipeline(package_path=PIPELINE_PACKAGE_PATH):
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=package_path,
    )
client = Client(host="http://localhost:8080",)

build_pipeline()

exp = client.create_experiment("new_experiment")
# PIPELINE_VERSION="2.0.0-rc.2"
# if (pipeline_id := client.get_pipeline_id(PIPELINE_NAME)) == None:
#     print("Pipeline creation ...")
#     client.upload_pipeline(
#         pipeline_package_path=PIPELINE_PACKAGE_PATH,
#         pipeline_name=PIPELINE_NAME,
#         description="An test pipeline.",
#     )

#     print("Initializing pipeline version ...")
#     client.upload_pipeline_version(
#         pipeline_name=PIPELINE_NAME,
#         pipeline_package_path=PIPELINE_PACKAGE_PATH,
#         pipeline_version_name=PIPELINE_VERSION_NAME_BASE + "_0",
#     )
# else:
#     pipeline_version_id = client.list_pipeline_versions(pipeline_id=pipeline_id, sort_by="pipeline_version_id desc").pipeline_versions[0].pipeline_version_id
#     pipeline_name = client.get_pipeline_version(pipeline_id=pipeline_id, pipeline_version_id=pipeline_version_id).display_name
#     last_version = pipeline_name.split("_")[-1]

#     client.upload_pipeline_version(
#         pipeline_id=pipeline_id,
#         pipeline_package_path=PIPELINE_PACKAGE_PATH,
#         pipeline_version_name=PIPELINE_VERSION_NAME_BASE + f"_{int(last_version) + 1}",
#     )

client.create_run_from_pipeline_package(
    pipeline_file=PIPELINE_PACKAGE_PATH,
    experiment_id=exp.experiment_id,
)

