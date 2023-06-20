#!/bin/sh
export $(xargs < env)

echo '################################### Package: Build, Upload to GCS ###################################'
python setup.py check
rm -rf dist/
python setup.py egg_info --egg-base /tmp sdist
PACKAGE_URI_DIR=$(python -c "import os; from my_package.config import get_package_uri; print(get_package_uri(dir_path=True))")
echo 
gsutil cp -r "dist/." "$PACKAGE_URI_DIR"
echo $PACKAGE_URI_DIR
echo 

echo '################################### Training pipeline: Compile, Upload to GCS, Run ###################################'
python -c "from pipeline.train import compile_pipeline; compile_pipeline()"
PIPELINE_TRAIN_PATH=$(python -c "from pipeline.train import pipeline_path; print(pipeline_path)")
echo 
gsutil cp "pipeline/training_pipeline.json" "$PIPELINE_TRAIN_PATH"
echo $PIPELINE_TRAIN_PATH
echo 
python -c "from pipeline.train import run_pipeline; run_pipeline()"
echo 

# echo '################################### Batch Prediction pipeline: Compile, Upload to GCS, Run/Schedule ###################################'
# python -c "from pipeline.predict import compile_pipeline; compile_pipeline()"
# PIPELINE_PREDICT_PATH=$(python -c "from pipeline.predict import pipeline_path; print(pipeline_path)")
# echo 
# gsutil cp "pipeline/prediction_pipeline.json" "$PIPELINE_PREDICT_PATH"
# echo $PIPELINE_PREDICT_PATH
# echo 
# python -c "from pipeline.predict import run_pipeline; run_pipeline()"
# echo 

# # # echo '################################### Online Prediction pipeline: Test predict function ###################################'
# # # python -c "from my_package.predict import predict"
# # # echo 

# # # # # Compipe prediction pipeline
# # # echo 
# # # echo '################################### Compile Batch Prediction Pipeline ###################################'
# # # python -c "from pipeline.predict import compile_pipeline; compile_pipeline()"
