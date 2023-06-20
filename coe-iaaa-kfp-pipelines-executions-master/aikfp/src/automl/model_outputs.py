from sklearn.metrics import log_loss,accuracy_score,roc_auc_score
import pandas as pd
import numpy as np
import catboost as ctb
def cls_metric_auc(yreal,ypred):
    if ypred.shape[1]<=2:return roc_auc_score(yreal,ypred[:,1])
    else:return roc_auc_score(yreal,ypred, multi_class='ovr', average='weighted')
def cls_feature_name(x,m):
    if m=='LGB':
        try:x= x.feature_name_
        except: x=x.feature_name()
    elif m=='XGB':x= x.get_booster().feature_names
    elif m=='CTB': x= x.feature_names_
    return [y.replace('#','_').replace('%','_').replace('[','_').replace(']','_').replace(' ','_') for y in x]
def cls_feature_importance(x,m):
    if m=='LGB': 
        try:rf=pd.DataFrame({'feature':x.feature_name_,'importance':x.feature_importances_})
        except: rf=pd.DataFrame({'feature':x.feature_name(),'importance':x.feature_importance()})
    elif m=='XGB':
        rf=pd.DataFrame(x.get_booster().get_score(importance_type="weight"),index=['importance']).T.reset_index().rename({'index':'feature'},axis=1)
        rf_=pd.DataFrame({'feature':x.get_booster().feature_names,'importance':0})
        rf=pd.concat([rf,rf_[~rf_['feature'].isin(rf['feature'])]])
    elif m =='CTB': rf=pd.DataFrame({'feature':x.feature_names_,'importance':x.get_feature_importance()})
    return rf.sort_values('importance', ascending=False).reset_index(drop=True)
def cls_predict_proba(m,x,t,r=None):
    if t=='LGB': return m.predict_proba(x,num_iteration=m.best_iteration_)
    elif t=='LGB_B':
        result=m.predict(x,num_iteration=m.best_iteration)
        if pd.DataFrame(result).shape[1]>1 :return result
        else: return np.vstack((1. - result, result)).transpose()    
    elif t=='XGB': return m.predict_proba(x,iteration_range=(0,m.best_iteration))
    elif t=='CTB':  return m.predict_proba(ctb.Pool(x, cat_features = r),ntree_start=0, ntree_end=m.get_best_iteration(),thread_count=-1)
    elif t=='RL':  return m.predict_proba(x)
def cls_loss_training(m,x,t):
    if t=='LGB':  df={'training':m.evals_result_['training'][x],'validation':m.evals_result_['valid_1'][x]}
    elif t=='XGB': df= {'training':m.evals_result()['validation_0'][x],'validation':m.evals_result()['validation_1'][x]}
    elif t=='CTB': df={'training':m.get_evals_result()['learn']['MultiClass'],'validation':m.get_evals_result()['validation']['MultiClass']}
    df=pd.DataFrame(df).reset_index().rename({'index':'iteration'},axis=1)
    return pd.melt(df,id_vars=['iteration'],value_vars=['training','validation'],var_name='metric')