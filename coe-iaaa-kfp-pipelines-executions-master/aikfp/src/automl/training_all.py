from sklearn.metrics import log_loss,accuracy_score,roc_auc_score
import numpy as np
from . import model_outputs as Z
from . import model_selection as M
from . import pipeline_logger as P
from . import model_building as B
def save_artifacts_and_logs(x_tr,y_tr,x_vl,y_vl,params,cfg,logger):
    model=B.model_method(x_tr,y_tr,x_vl,y_vl,params,cfg)
    yt=Z.cls_predict_proba(model,x_tr,cfg['type_model'],cfg['cat_features'])
    yv=Z.cls_predict_proba(model,x_vl,cfg['type_model'],cfg['cat_features'])
    # cfg['feature_importance']=cls_feature_importance(model,cfg['type_model'])
    _log_loss=log_loss(y_tr, yt),log_loss(y_vl, yv)
    _accuracy_score=accuracy_score(y_tr,np.argmax(yt ,axis=1)),accuracy_score(y_vl,np.argmax(yv ,axis=1))
    _roc_auc_score=Z.cls_metric_auc(y_tr,yt),Z.cls_metric_auc(y_vl,yv)
    P.save_system_line(logger,f'Training -- log_loss: {_log_loss[0]}, accuracy: {_accuracy_score[0]}, roc_auc_score: {_roc_auc_score[0]}')
    P.save_system_line(logger,f'Validation -- log_loss: {_log_loss[1]}, accuracy: {_accuracy_score[1]}, roc_auc_score: {_roc_auc_score[1]}')
    return {'model':model,
    'data_metrics': {
        'training':{ 'y_real': y_tr, 'y_predict': yt , 'y_predict_label': np.argmax(yt,axis=1)},
        'validation':{ 'y_real': y_vl, 'y_predict': yv , 'y_predict_label': np.argmax(yv,axis=1)},
        },
    'metrics':{
        'training':{ 'log_loss': _log_loss[0], 'accuracy':_accuracy_score[0],'roc_auc_score':_roc_auc_score[0], 'gini': 2*_roc_auc_score[0]-1},
        'validation':{ 'log_loss': _log_loss[1], 'accuracy': _accuracy_score[1],'roc_auc_score':_roc_auc_score[1], 'gini': 2*_roc_auc_score[1]-1},
    }}
