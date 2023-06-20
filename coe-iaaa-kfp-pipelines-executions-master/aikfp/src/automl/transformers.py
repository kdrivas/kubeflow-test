from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,RobustScaler,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from aiutils import aiwr
import pandas as pd
import numpy as np
from .utils import exist_element_cfg 
from .pipeline_logger import save_system_line
# ##########################################################
# ##################### PreProcessors ######################
# ##########################################################

# class CastToFloatAndCategory(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.cast_to_number =[]
#         self.cast_to_category = []
        
#     def fit(self, X, y=None):
#         for col, dtype in X.dtypes.iteritems():
#             dtype = str(dtype)
#             if dtype[:3] in ['Int', 'int', 'flo']:
#                 self.cast_to_number.append(col)
#             elif dtype in ['string', 'category']:
#                 self.cast_to_category.append(col)
#             elif dtype == 'object':
#                 try:
#                     X[col].astype(object).fillna(np.nan).astype(np.float32)
#                     self.cast_to_number.append(col)
#                 except:
#                     X[col].astype('category')
#                     self.cast_to_category.append(col)
#         return self
        
#     def transform(self, X):
#         X = X.copy()
#         for col in self.cast_to_number:
#             X[col] = X[col].astype(np.float32)
#         for col in self.cast_to_category:
#             X[col] = X[col].astype('category')
#         return X

##########################################################
######################## Encoders ########################
##########################################################

class MultipleLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder for multiple columns, not taking the nulls
    
    Args:
        df (pd.DataFrame): Pandas DataFrame.
        columns (list, optional): Columns of the DataFrame, default takes
            object columns.
        suffix (str, optional): Suffix of encoded features, default inplace
            in same columns.
        handle_unknown (bool, optional): Where to ignore (True) or not
            new values when applying fitted encoder.
            
    Examples:
        - Inplace all columns:
            mle = aiwr.MultipleLabelEncoder()
            df = mle.fit_transform(df)
        - Encode some columns into "_encoded" features:
            mle = aiwr.MultipleLabelEncoder(columns=['CUC', 'DISTRITO'],
                                            suffix='_encoded')
            df = mle.fit_transform(df)
    """
    def __init__(self, columns=None, exclude=None, suffix='', handle_unknown=True):
        self.columns = aiwr._define_columns(columns)
        self.exclude = aiwr._define_columns(exclude)
        self.dict_le = {}
        self.suffix = suffix
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        # persist label encoders in dictionary
        if self.columns is None:
            self.columns = aiwr.dict_dtypes(X)['cat']
        if self.exclude:
            self.columns = [col for col in self.columns if col not in self.exclude]
        for feature in self.columns:
            le = LabelEncoder()
            le.fit(list(X[feature].dropna()))
            self.dict_le[feature] = le
        return self

    def transform(self, X):
        X = X.copy()
        for feature, le in self.dict_le.items():
            feature_encode = feature + self.suffix
            if self.handle_unknown:
                idx = X[feature].isin(list(le.classes_))
                X.loc[~idx, feature_encode] = np.nan
                X.loc[ idx, feature_encode] = le.transform(X.loc[idx, feature])
            else:
                X[feature_encode] = le.transform(X[feature])
            X[feature_encode] = X[feature_encode].astype(float)
        return X

# class OneHotEncoder2(BaseEstimator, TransformerMixin):
#     """Get dummies of a DataFrame and the columns argument.
    
#     Args:
#         df (pd.DataFrame): Pandas DataFrame.
#         columns (list, optional): Columns of the DataFrame, defualt takes
#             all columns.
#         concat (bool, optional): Whether to return the dummies appended to
#             the DataFrame or not. Default True.
#         handle_unknown (str, optional): Whether to raise an error or ignore
#             if an unknown categorical feature is present during transform.
#             When this parameter is set to 'ignore' and an unknown category
#             is encountered during transform, the resulting one-hot encoded
#             columns for this feature will be all zeros. In the inverse
#             transform, an unknown category will be denoted as None.
#         fillna (str, optional): Value to use to fill nan values in columns.
#         kargs (optional): Additional arguments for the OneHotEncoder() function
    
#     Examples:
#         - df, ohe = aiwr.dummies(df, ['ESTADO_CIVIL', 'GRADO_INSTRUCCION'])
    
#     Returns:
#         OneHotEncoder2 class:
#             - Pandas DataFrame merged with dummies if concat,
#                 else returns the DataFrame of only the dummies
#             - OneHotEncoder fitted by DataFrame and Columns
#     """
    
#     def __init__(self, columns=None, concat=True, handle_unknown='ignore', fillna='nan', **kwargs):
#         self.columns = _define_columns(columns)
#         self.concat = concat
#         self.handle_unknown = handle_unknown
#         self.fillna = fillna
#         self.kwargs = kwargs
        
#     def fit(self, X, y=None):
#         if self.columns is None:
#             self.columns = list(X.columns)
#         ohe = OneHotEncoder(sparse=False, handle_unknown=self.handle_unknown, **self.kwargs)
#         result_array = ohe.fit(X[self.columns].fillna(self.fillna))
#         self.result_columns = ohe.get_feature_names(self.columns)
#         self.ohe = ohe
#         return self
        
#     def transform(self, X):
#         result_array = self.ohe.transform(X[self.columns].fillna(self. fillna))
#         result = pd.DataFrame(result_array, index=X.index, columns=self.result_columns, dtype=np.int8)
#         if self.concat:
#             return pd.concat([X, result], axis=1)
#         else:
#             return result
def label_encoder(df, cols_cat, suffix='', inplace=False):
    df0 = df if inplace else df.copy()
    dict_le = {}
    for var in cols_cat:
        le = LabelEncoder()
        le.fit(list(df0[var].dropna()))
        dict_le[var] = le
    return dict_le

class transformer_skip(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.c=1
    def fit(self, df,y = None):
        return self
    def transform(self, df,y=None):
        return df      
class transformer_cat_null(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.c=1    
    def fit(self, X,y = None):
        return self
    def transform(self, X,y=None):
        for var in X.columns: 
            X[var].fillna('missing_value',inplace=True)   
            X[var] = X[var].apply(lambda x: 'missing_value' if str(x).lower() in ['nan','none','nd','nat'] else x)
        return X
    
class transformer_cat_type(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.c=1
    def fit(self, X,y = None):
        tmp=X.copy()
        n_el=tmp[X.columns].nunique().max()+1
        self.c={}
        for x in X.columns:
            self.c[x]=list(tmp[x].unique())+list(np.repeat('missing_value',n_el-tmp[x].nunique()))
            self.c[x]=self.c[x][:n_el]
        self.c=pd.DataFrame(self.c)
        return self
    def transform(self, X,y=None):
        for var in X.columns:
            intersection=set(X[var].unique()).intersection(self.c[var])
            try:X[var] = self.dict_le[var].transform(X[var].apply(lambda x: 'missing_value' if x not in list(intersection) else x))
            except:
                CTE=X[var].value_counts().index[0]
                X[var] = X[var].apply(lambda x: CTE if x not in list(set(intersection).difference(['missing_value'])) else x)
                X[var]=X[var].astype('category')
        return X
class transformer_cat_le(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dict_le = {}
    def fit(self, X,y = None):
        tmp=X.copy()
        n_el=tmp[X.columns].nunique().max()+1
        self.c={}
        for x in X.columns:
            self.c[x]=list(tmp[x].unique())+list(np.repeat('missing_value',n_el-tmp[x].nunique()))
            self.c[x]=set(self.c[x][:tmp[x].nunique()])
        # self.c=pd.DataFrame(self.c)
        self.dict_le = label_encoder(X, X.columns, suffix = "", inplace=False)
        return self
    def transform(self, X,y=None):
        for var in X.columns:
            intersection=set(X[var].unique()).intersection(self.c[var])
            try:
                CTE=X[var].value_counts().index[0]
                X[var] = self.dict_le[var].transform(X[var].apply(lambda x: CTE if x not in list(set(intersection).difference(['missing_value'])) else x))                   
            except:                
                X[var] = self.dict_le[var].transform(X[var].apply(lambda x: 'missing_value' if x not in list(intersection) else x))             
        return X
class transformer_cat_ohe(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.name_columns = []
    def fit(self, X,y = None):
        tmp=X.copy()
        n_el=tmp[X.columns].nunique().max()+1
        self.c={}
        for x in X.columns:
            self.c[x]=list(tmp[x].unique())+list(np.repeat('missing_value',n_el-tmp[x].nunique()))
            self.c[x]=self.c[x][:n_el]
        # self.c=pd.DataFrame(self.c)
            intersection=set(X[x].unique()).intersection(self.c[x])
            try:X[x] = X[x].apply(lambda x: 'missing_value' if x not in list(intersection) else x)
            except:
                CTE=X[x].value_counts().index[0]
                X[x] = X[x].apply(lambda x: CTE if x not in list(set(intersection).difference(['missing_value'])) else x)
        self.name_columns=pd.get_dummies(X,dummy_na=True,prefix_sep='_ohe_').columns.to_list() 
        self.name_columns=[x.replace('#','_').replace('%','_').replace('[','_').replace(']','_').replace(' ','_') for x in self.name_columns]
        return self
    def transform(self, X,y=None): 
        X=pd.get_dummies(X[list(set([ x.split('_ohe_')[0] for x in self.name_columns]))],dummy_na=True,prefix_sep='_ohe_')
        X.columns=[x.replace('#','_').replace('%','_').replace('[','_').replace(']','_').replace(' ','_') for x in X.columns]
        for y in list(set(self.name_columns).difference(X.columns)):
            X[y]=0
            print(y)
        X.columns=[x.replace('#','_').replace('%','_').replace('[','_').replace(']','_').replace(' ','_') for x in X.columns]
        return X[self.name_columns]
class transformer_cat_pareto(BaseEstimator, TransformerMixin):
    def __init__(self,pareto_treshold):
        self.pareto_treshold=pareto_treshold
    def fit(self, X,y = None):
        self.pareto_dc={}
        for x in X.columns:
            p = (X[x].value_counts(normalize=True)).cumsum()
            p=p[p < self.pareto_treshold].index.to_list()
            self.pareto_dc[x]=p
        return self
    def transform(self, X,y=None):
        for x in X.columns:
            X.loc[~(X[x].isin(self.pareto_dc[x]+['missing_value'])),x] = 'OTROS'
        return X
class transformer_num_remove_outliers(BaseEstimator, TransformerMixin):
    def __init__(self,outlier_range):
        self.c={}
        self.outlier_range=outlier_range
    def fit(self,X,y=None):
        for x in X.columns:
            Q1 = np.percentile(X[x], 25)
            Q3 = np.percentile(X[x],75)
            IQR = Q3 - Q1
            self.c[x]=[Q1-self.outlier_range*IQR,Q3+self.outlier_range*IQR]
        return self
    def transform(self, X, y =None):
        for x in X.columns:
            min_value,max_value=self.c[x]
            X.loc[X[x]<min_value, x] = min_value
            X.loc[X[x]>max_value, x] = max_value
        return X    

def preprocess(df,cfg,logger):
    exist_element_cfg(cfg,'steps_cat_preprocess',['null','pareto','le'])
    exist_element_cfg(cfg,'steps_cont_preprocess',['skip'])
    exist_element_cfg(cfg['preprocess_constants'],'num_fill_null','median')
    exist_element_cfg(cfg['preprocess_constants'],'cat_pareto',0.8)
    exist_element_cfg(cfg['preprocess_constants'],'num_outlier_range',1.5)
    preprocess_dc={
     'categoric_skip': ('categoric_skip', transformer_skip()),
    'categoric_null': ('categoric_null', transformer_cat_null()),
    'categoric_pareto':('categoric_pareto',transformer_cat_pareto(cfg['preprocess_constants']['cat_pareto'])),
    'categoric_le':('categoric_le',transformer_cat_le()),
    'categoric_ohe':('categoric_ohe',transformer_cat_ohe()),
    'categoric_type':('categoric_type',transformer_cat_type()),
         'numeric_skip': ('categoric_skip', transformer_skip()),
     'numeric_null': ('numeric_null', SimpleImputer(strategy = cfg['preprocess_constants']['num_fill_null'])),
    'numeric_ss':  ('numeric_ss', StandardScaler()),
     'numeric_rs':   ('numeric_rs', RobustScaler()),
      'numeric_ms':  ('numeric_ms', MinMaxScaler()),
         'numeric_outlier':  ('numeric_outlier', transformer_num_remove_outliers(cfg['preprocess_constants']['num_outlier_range'])),
                    }
    cat_preprocess = Pipeline( steps = [preprocess_dc[f'categoric_{x}'] for x in cfg['steps_cat_preprocess']], verbose=True) 
    cont_preprocess = Pipeline( steps = [preprocess_dc[f'numeric_{x}'] for x in cfg['steps_cont_preprocess']], verbose=True) 
    preprocessor = ColumnTransformer([
        ('cat', cat_preprocess, cfg['cat_features']),
        ('num', cont_preprocess, cfg['cont_features']),
    ],remainder='passthrough')
    df_transform = preprocessor.fit_transform(df)
    if 'ohe' in cfg['steps_cat_preprocess']:cfg['cat_features']=preprocessor.named_transformers_['cat']['categoric_ohe'].name_columns
    cfg['name_features']=cfg['cat_features']+cfg['cont_features']
    df_transform=pd.DataFrame(df_transform,columns=cfg['name_features'])
    save_system_line(logger,f"Tipo_Preprocess: cat - {cfg['steps_cat_preprocess']}, cont - {cfg['steps_cont_preprocess']}")
    return df_transform,preprocessor