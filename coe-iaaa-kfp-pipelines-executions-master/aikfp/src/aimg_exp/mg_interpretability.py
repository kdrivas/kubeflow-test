import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from aikfp.src.automl import model_outputs as Z
from aikfp.src.aimg_exp import querys as Q
def exp_feature_explainer(x_test,y_test,sub_cols,svv,model,preprocess,obj,filter_,_vars):
    features=filter_.name        
    _vars=[x for x in _vars if x!='nan']
    def report_shap(tmp,_mf,mode='all',fmode='all'):
        df=pd.DataFrame({'feature':svv.feature_names})
        df[features]=fmode
        for i,v in enumerate(obj['sub_cols']):
            shap_df = pd.DataFrame(svv.values[:,:,i], columns=svv.feature_names).reset_index(drop=True).iloc[tmp.index]
            df[v] = np.abs(shap_df.values).mean(0)
        df['all']=df[obj['sub_cols']].mean(axis=1)
        df=pd.melt(df,id_vars=['feature']+[features],value_vars=obj['sub_cols']+['all'],var_name='name_classes')
        df['type_metric']=f'explainer_by_{mode}'
        df['metric']='shap_abs'
        _mf.append(df)
        return _mf    
    def report_linear(tmp,_mf,mode='all',fmode='all'):
        def calculate_correlation(x_test,w,var):
            x_test=x_test.reset_index(drop=True).iloc[tmp.index]
            w=w.reset_index(drop=True).iloc[tmp.index]
            mf=pd.DataFrame(pd.concat([pd.DataFrame(preprocess.transform(x_test),columns=obj['name_features']),w],axis=1).corr().loc[var]).abs().reset_index().rename({'index':'feature',var:'value'},axis=1)
            mf['type_metric']=f'explainer_by_{mode}'
            mf[features]=fmode
            mf['metric']='linear'
            if var==obj['name_target']:mf['name_classes']='all'
            else:mf['name_classes']=var
            return mf       
        _mf.append(calculate_correlation(x_test,y_test,obj['name_target']))
        for y in obj['sub_cols']:_mf.append(calculate_correlation(x_test,sub_cols[y],y))
        return _mf
    df_shap,df_linear=[],[]
    df_shap=report_shap(x_test.reset_index(drop=True),df_shap,'all','all')
    df_linear=report_linear(x_test.reset_index(drop=True),df_linear,'all','all')
    filter_=filter_.reset_index(drop=True)
    for x in _vars:
        tmp=filter_[filter_==x].copy()
        df_shap=report_shap(tmp,df_shap,'subpoblation',x)
        df_linear=report_linear(tmp,df_linear,'subpoblation',x)
    df_gn=Z.cls_feature_importance(model,obj['type_model']).rename({'importance':'value'},axis=1)
    df_gn['type_metric']='explainer_by_all'
    df_gn['metric']='gain'
    df_gn['name_classes']='all'
    df_gn[features]='all'
    best_features=df_gn.sort_values('value',ascending=False)[:10]['feature']
    _xf=[]
    for n,v in enumerate(obj['sub_cols']):
        xf=pd.DataFrame(svv.values[:,:,n],columns=svv.feature_names)[best_features].rename(dict(zip(best_features,[f'feature_{x}' for x in range(10)])),axis=1)
        xf['name_classes']=v
        _xf.append(xf)
    if obj['source']=='kf_training':rf=pd.concat([pd.concat(df_shap),pd.concat(df_linear),df_gn])
    elif obj['source']=='kf_predict':rf=pd.concat([pd.concat(df_shap),pd.concat(df_linear)])
    return {'feature__explainer_general':Q.bq_generate_and_save(rf,
                                                       'mg_feature__explainer',obj
                                                       ),
            'best_features':best_features,
            'feature__explainer_individual':Q.bq_generate_and_save(pd.concat(_xf),
                                                       'mg_feature__explainer_individual',obj
                                                       )}