import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline, make_pipeline
def model_method(x_tr,y_tr,x_vl,y_vl,params,cfg):
    if (cfg['type_model']=='LGB') & (cfg['framework_model']=='TABULAR'):
        params['n_jobs']=1
        params['n_estimators']=cfg['cfg_parameters']['n_estimators']
        return lgb.LGBMClassifier(**params).fit(x_tr,y_tr,
                                    eval_set=[(x_tr, y_tr),(x_vl, y_vl)],
                                    eval_metric=cfg['metric'], 
                                    categorical_feature=[x_tr.columns.to_list().index(i) for i in cfg['cat_features']],
                                    callbacks=[lgb.early_stopping(stopping_rounds=cfg['cfg_parameters']['early_stopping_rounds'], verbose=200)]
                                               ) 
    elif (cfg['type_model']=='XGB') & (cfg['framework_model']=='TABULAR'): 
        params['n_jobs']=1
        params['objective']=cfg['obj']
        params['num_class'] = cfg['n_classes']
        params['n_estimators']=cfg['cfg_parameters']['n_estimators']
        params['use_label_encoder']=False
        # params['enable_categorical']=True
        return xgb.XGBClassifier(**params).fit(x_tr,y_tr,
                                    eval_set=[(x_tr, y_tr),(x_vl, y_vl)],
                                    early_stopping_rounds=cfg['cfg_parameters']['early_stopping_rounds'],
                                    # enable_categorical=True,
                                    eval_metric=cfg['metric'],
                                    verbose=200)
    elif (cfg['type_model']=='CTB') & (cfg['framework_model']=='TABULAR'):
        params['objective']=cfg['obj']
        params['eval_metric']=cfg['metric']
        pool_tr = ctb.Pool(x_tr, y_tr, cat_features = cfg['cat_features'])
        pool_vl = ctb.Pool(x_vl, y_vl, cat_features = cfg['cat_features'])
        return ctb.CatBoostClassifier(**params).fit(pool_tr, eval_set=[pool_vl], early_stopping_rounds=cfg['cfg_parameters']['early_stopping_rounds'], verbose=200)
    elif (cfg['type_model']=='RL') & (cfg['framework_model']=='NLP'):
        params['max_iter']=cfg['cfg_parameters']['n_estimators']
        params['n_jobs']=-1
        return make_pipeline(Pipeline(
                        steps = [
                            ('vect', CountVectorizer(analyzer = "word",min_df=cfg['cfg_parameters']['min_df'])),
                            ('tfidf', TfidfTransformer(norm = 'l2',
                                                       use_idf = cfg['cfg_parameters']['use_idf'],
                                                       sublinear_tf = cfg['cfg_parameters']['sublinear_tf'])),
                        ], verbose=True),
                            LogisticRegression(**params)).fit(x_tr,y_tr)