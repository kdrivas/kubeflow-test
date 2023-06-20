from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold,GroupKFold,train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
def selection_method(X_train, y_train,cfg):    
    if cfg['selection_method']=='stratified':return train_test_split(X_train, y_train, random_state=cfg['cfg_parameters']['seed'],test_size=cfg['cfg_parameters']['test_size'],stratify=y_train)
    elif cfg['selection_method']=='normal':return train_test_split(X_train, y_train, random_state=cfg['cfg_parameters']['seed'],test_size=cfg['cfg_parameters']['test_size'])
    elif cfg['selection_method']=='undersampling':return split_and_undersampling(X_train, y_train, {'test_size':cfg['cfg_parameters']['test_size'],'seed':cfg['cfg_parameters']['seed']})
    elif cfg['selection_method']=='timeseries':return split_over_timeseries(X_train, y_train, {'test_size':cfg['cfg_parameters']['test_size'],'seed':cfg['cfg_parameters']['seed']})
    elif cfg['selection_method']=='group':return split_over_groups(X_train,y_train, {'group':cfg['group'],'seed':cfg['cfg_parameters']['seed']})
def kfold_selection_method(x_train,y_train,cfg):
    cfg['strategy_fold']='use_folds'
    if cfg['selection_method']=='normal': return KFold(n_splits=cfg['n_folds']).split(x_train)
    elif cfg['selection_method']=='stratified': return StratifiedKFold(n_splits=cfg['n_folds']).split(x_train,y_train)
    elif cfg['selection_method']=='timeseries':return TimeSeriesSplit(n_splits=cfg['n_folds']).split(x_train)
    elif cfg['selection_method']=='group':
        cfg['n_folds']=cfg['group'].nunique()
        return GroupKFold(n_splits=cfg['n_folds']).split(x_train,y_train,cfg['group'])
    elif cfg['selection_method']=='undersampling':return split_and_undersampling_w_folds(x_train,y_train,cfg)
def split_and_undersampling(X_train,y_train,cfg,c_min=0.1):
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_train, y_train, test_size=cfg['test_size'], random_state=cfg['seed'],stratify=y_train) 
    a = y_tr.value_counts().values[1]
    b = int((a / c_min) - a)
    rus = RandomUnderSampler(random_state=cfg['seed'], sampling_strategy={0: b, 1: a})
    x_sample, y_sample = rus.fit_resample(X_tr, y_tr.ravel())
    X_tr = X_tr.iloc[rus.sample_indices_].copy()
    y_tr = y_tr.iloc[rus.sample_indices_].copy()
    return X_tr, X_vl, y_tr, y_vl 
def split_and_undersampling_w_folds(X_train,y_train,cfg,c_min=0.1):
    skf=StratifiedKFold(n_splits=cfg['n_folds']).split(X_train,y_train)
    skf_list=[]
    for fold,(idx_tr,idx_vl) in enumerate(list(skf)):
        X_tr, X_vl=X_train.iloc[idx_tr],X_train.iloc[idx_vl]
        y_tr, y_vl=y_train.iloc[idx_tr],y_train.iloc[idx_vl] 
        a = y_tr.value_counts().values[1]
        b = int((a / c_min) - a)
        rus = RandomUnderSampler(random_state=cfg['cfg_parameters']['seed'], sampling_strategy={0: b, 1: a})
        x_sample, y_sample = rus.fit_resample(X_tr, y_tr.ravel())
        X_tr = X_tr.iloc[rus.sample_indices_].copy()
        y_tr = y_tr.iloc[rus.sample_indices_].copy()
        skf_list.append((np.array(X_tr.index.tolist()),np.array(X_vl.index.tolist())))
    return skf_list 
def split_over_timeseries(X_train,y_train,cfg):
    n=int((1-cfg['test_size'])*len(X_train))
    return X_train.iloc[:n],X_train.iloc[n:],y_train.iloc[:n],y_train.iloc[n:] 
def split_over_groups(X_train,y_train,cfg):
    ct=[cfg['group'].max(),cfg['group'].min()]
    return X_train.iloc[cfg['group'][~cfg['group'].isin(ct)].index],X_train.iloc[cfg['group'][cfg['group'].isin(ct)].index],y_train.iloc[cfg['group'][~cfg['group'].isin(ct)].index],y_train.iloc[cfg['group'][cfg['group'].isin(ct)].index]
def fix_selection_method(x_tr,y_tr,x_vl,y_vl,df,cols):
    _id=df[cols]
    name=list(set(cols).difference(['periodo']))
    id_tr,id_vl=_id.iloc[x_tr.index],_id.iloc[x_vl.index]
    id_tr,id_vl=id_tr.drop_duplicates(keep='last'),id_vl.drop_duplicates(keep='last')
    index=list(set(id_vl[name[0]]).difference(id_tr[name[0]]))
    id_vl=id_vl[id_vl[name].isin(index)]
    x_tr,x_vl=x_tr.loc[id_tr.index],x_vl.loc[id_vl.index.tolist()]
    y_tr,y_vl=y_tr.loc[id_tr.index],y_vl.loc[id_vl.index.tolist()]
    return x_tr,y_tr,x_vl,y_vl