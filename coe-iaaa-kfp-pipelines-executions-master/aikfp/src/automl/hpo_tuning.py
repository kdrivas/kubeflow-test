import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import log_loss
from . import pipeline_logger as P
def hpo_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger):
    if (cfg['type_model']=='LGB')&(cfg['framework_model']=='TABULAR'): return lgb_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger)
    elif (cfg['type_model']=='XGB')&(cfg['framework_model']=='TABULAR'): return xgb_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger)
    elif (cfg['type_model']=='CTB')&(cfg['framework_model']=='TABULAR'): return ctb_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger)
    elif (cfg['type_model']=='RL')&(cfg['framework_model']=='NLP'): return rl_nlp_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger)


def lgb_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger):        
    max_depth=trial.suggest_int('max_depth', 3, 9)
    params = {
'objective':trial.suggest_categorical("objective", [cfg['obj']]),
'metric':trial.suggest_categorical("metric", [cfg['metric']]),
'learning_rate':trial.suggest_categorical("learning_rate", [cfg['cfg_parameters']['l_rate']]),
'n_jobs':trial.suggest_categorical("n_jobs", [-1]),
'verbose':trial.suggest_categorical("verbose", [-1]),
        # "class_weight": trial.suggest_categorical('class_weight',[None, "balanced"]),
 'num_classes':trial.suggest_categorical("num_classes", [cfg['n_classes']]),
 'random_state':trial.suggest_categorical("random_state", [cfg['cfg_parameters']['seed']]),
   'bagging_seed':trial.suggest_categorical("bagging_seed", [cfg['cfg_parameters']['seed']]),
 'force_col_wise':trial.suggest_categorical("force_col_wise", [True]),
            'num_leaves':trial.suggest_int('num_leaves', 2, np.power(2,max_depth)),     
         # 'min_data_in_leaf':trial.suggest_int('min_data_in_leaf', 1, 1e3),
         'min_child_samples':trial.suggest_int('min_child_samples', 0, 1e4),
        'min_child_weight':trial.suggest_float('min_child_weight', 1e-5, 1e4),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-1, 1e2, log=True),
        # 'bagging_fraction':trial.suggest_float('bagging_fraction',1e-3, 1),
 'subsample':trial.suggest_float('subsample', 1e-1, 1.0,log=True),     
'colsample_bytree':trial.suggest_float('colsample_bytree', 0, 1.0),
'reg_alpha':trial.suggest_float('reg_alpha', 1e-5, 1e2, log=True),
'reg_lambda':trial.suggest_float('reg_lambda', 1e-5, 1e2, log=True),
'max_bin':trial.suggest_int('max_bin', 2, 5e2),
    'max_cat_threshold':trial.suggest_int('max_cat_threshold', 1, 1e2),
    'path_smooth':trial.suggest_float('path_smooth', 1, 1e3),
    'cat_l2':trial.suggest_float('cat_l2', 0, 1e3),
    'cat_smooth':trial.suggest_float('cat_smooth', 0, 1e3),
    'max_cat_to_onehot':trial.suggest_int('max_cat_to_onehot', 1, 1e2),
# 'categorical_feature': trial.suggest_categorical('categorical_feature',[x_tr.columns.to_list().index(i) for i in cfg['cat_features']])
}
    params['max_depth']=max_depth
    model = lgb.train(params,
                      train_set=lgb.Dataset(x_tr, label=y_tr,
                                            categorical_feature=[x_tr.columns.to_list().index(i) for i in cfg['cat_features']]),
                      valid_sets=lgb.Dataset(x_vl, label=y_vl,
                                             categorical_feature=[x_tr.columns.to_list().index(i) for i in cfg['cat_features']]),
                      num_boost_round =cfg['cfg_parameters']['n_estimators'],

                       callbacks=[optuna.integration.LightGBMPruningCallback(trial, cfg['metric'], valid_name='valid_0'),
                                 lgb.early_stopping(stopping_rounds=cfg['cfg_parameters']['early_stopping_rounds'],verbose=False)])
    metric=log_loss(y_vl,model.predict(x_vl,num_iteration=model.best_iteration))
    P.save_system_line(logger,f'trial: {trial.number} val_logloss: {metric}')
    return metric
def xgb_objective( trial,x_tr,y_tr,x_vl,y_vl,cfg,logger):
    max_depth=trial.suggest_int('max_depth', 2, 8)
    params = { 
        'tree_method':trial.suggest_categorical("tree_method", ['hist']),#auto, gpu_hist
        'objective':trial.suggest_categorical("objective", [cfg['obj']]),
        'eval_metric':trial.suggest_categorical("eval_metric", [cfg['metric']]),
        'num_class':trial.suggest_categorical("num_class", [cfg['n_classes']]),
        'learning_rate':trial.suggest_categorical("learning_rate", [cfg['cfg_parameters']['l_rate']]),
        'grow_policy':trial.suggest_categorical("grow_policy", ["depthwise",'lossguide']),
        'random_state':trial.suggest_categorical("random_state", [cfg['cfg_parameters']['seed']]),
         'n_jobs':trial.suggest_categorical("n_jobs",[-1]),
        'min_child_weight':trial.suggest_float('min_child_weight', 1e-5, 1e4),
        'max_delta_step':trial.suggest_int('max_delta_step', 0, 30),
        # 'scale_pos_weight':trial.suggest_float('scale_pos_weight',0,1),
        'subsample':trial.suggest_float('subsample', 0.3, 1.0, log=True),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0, 1.0),
        'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0, 1.0),
        'colsample_bynode':trial.suggest_float('colsample_bynode', 0, 1.0),
        'max_leaves':trial.suggest_int('max_leaves', 1, np.power(2,max_depth)),
        'max_bin':trial.suggest_int('max_bin', 2, 5e2),
        'alpha':trial.suggest_float('alpha', 1e-5, 1e2,log=True),
        'lambda':trial.suggest_float('lambda', 1e-5, 1e2,log=True),
        'gamma':trial.suggest_float('gamma', 1e-5, 1e2,log=True),          
              }  
    params['max_depth']=max_depth
    #xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
    model = xgb.train(params,
                          dtrain=xgb.DMatrix(x_tr, y_tr,enable_categorical=True)  ,
                          evals=[(xgb.DMatrix(x_vl, y_vl,enable_categorical=True), 'valid')],
                          num_boost_round =cfg['cfg_parameters']['n_estimators'],verbose_eval=False,
                           callbacks=[
                               optuna.integration.XGBoostPruningCallback(trial, f"valid-{cfg['metric']}"),
                                      xgb.callback.EarlyStopping(rounds=cfg['cfg_parameters']['early_stopping_rounds'])])
    metric=log_loss(y_vl,model.predict(xgb.DMatrix(x_vl,enable_categorical=True), iteration_range=(0, model.best_iteration)))
    P.save_system_line(logger,f'trial: {trial.number} val_logloss: {metric}')
    return metric
def ctb_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger):
    max_depth=trial.suggest_int('depth', 2, 6)
    params = {
                    'eval_metric':trial.suggest_categorical('eval_metric',[cfg['obj']]),
                    # 'task_type':trial.suggest_categorical('task_type',['GPU']),
                    'objective':trial.suggest_categorical('objective',[cfg['metric']]),
                    'learning_rate':trial.suggest_categorical('learning_rate',[cfg['cfg_parameters']['l_rate']]),
                    'iterations':trial.suggest_categorical('iterations',[cfg['cfg_parameters']['n_estimators']]),
                    'random_seed':trial.suggest_categorical('random_seed',[cfg['cfg_parameters']['seed']]),
                    'bootstrap_type':trial.suggest_categorical('bootstrap_type',['Bernoulli']),#Bayesian
                    'use_best_model':trial.suggest_categorical('use_best_model',[True]),
                    'one_hot_max_size':trial.suggest_int('one_hot_max_size', 1, 20),
                    # 'max_leaves':trial.suggest_int('max_leaves', 1, np.power(2,max_depth)),#grow_policy
                   'subsample': trial.suggest_float('subsample',4e-2,1),
                   'max_ctr_complexity':trial.suggest_int('max_ctr_complexity', 1, 6),
                   'random_strength':trial.suggest_int('random_strength', 1, 6),
                    'rsm':trial.suggest_float('rsm', 5e-3, 1),
                    # "bagging_temperature":trial.suggest_float('bagging_temperature', 1, 10),
                    "reg_lambda":trial.suggest_float('reg_lambda', 1e-1, 1e3),
                   "min_data_in_leaf":trial.suggest_int('min_data_in_leaf', 1, 1e3),
                   "od_wait":trial.suggest_int('od_wait', 15, 35),
    }
    params['depth']=max_depth
    pool_tr = ctb.Pool(x_tr, y_tr, cat_features = cfg['cat_features'])
    pool_vl = ctb.Pool(x_vl, y_vl, cat_features = cfg['cat_features'])
    model = ctb.CatBoostClassifier(**params)
    model.fit(pool_tr, eval_set=[pool_vl], verbose=0, early_stopping_rounds=cfg['cfg_parameters']['early_stopping_rounds'])
    metric=log_loss(y_vl,model.predict_proba(pool_vl,ntree_start=0, ntree_end=model.get_best_iteration(),thread_count=-1))
    P.save_system_line(logger,f'trial: {trial.number} val_logloss: {metric}')
    return metric
def rl_nlp_objective(trial,x_tr,y_tr,x_vl,y_vl,cfg,logger):
    max_depth=trial.suggest_int('depth', 2, 6)
    params = {'max_iter':trial.suggest_categorical('max_iter',[cfg['cfg_parameters']['n_estimators']]),
              'n_jobs':trial.suggest_categorical('n_jobs',[-1]),
                    "C":trial.suggest_float('C', 1e-10, 1e3,log=True),
              "class_weight":trial.suggest_categorical('class_weight',["balanced",None]),
    }    
    model=make_pipeline(Pipeline(
                        steps = [
                            ('vect', CountVectorizer(analyzer = "word",min_df=cfg['cfg_parameters']['min_df'])),
                            ('tfidf', TfidfTransformer(norm = 'l2',
                                                       use_idf = cfg['cfg_parameters']['use_idf'],
                                                       sublinear_tf = cfg['cfg_parameters']['sublinear_tf'])),
                        ], verbose=True),
                            LogisticRegression(**params))
    model.fit(x_tr,y_tr)
    metric=log_loss(y_vl,model.predict_proba(x_vl))
    P.save_system_line(logger,f'trial: {trial.number} val_logloss: {metric}')
    return metric
