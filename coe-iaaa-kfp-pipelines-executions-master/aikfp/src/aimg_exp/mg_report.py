import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import os
colors=['gray','black','blue','green','pink','red','brown','darkred','yellow','orange','purple','salmon','black','black','black','black','black','black','black','black','black','black','black','black','black','black','black']
sns.set(font_scale = 1)
def plot_eda_distributions(rf,cols,mg_output,load_var,name_var,name_fields=['train','valid'],pal='mako'):
    """
    EDA
    Args:
        rf (pd.DataFrame): Pandas DataFrame.
        cols (dictionary): Relación de variables categóricas y numéricas.
        mg_output: Ruta del directorio.
        load_var: Nombre de Variable
        name_var: Nombre de Target.
        name_fields (list): Categorías del Target.      
    Examples:
        mg=MG_Artifacts_Cls(df,df_test,model,preprocess,artifacts,metadata)
        mg.get_all_artifacts('kf_predict')
    Returns:
        Plots - TimeSeries, Absolute Distribution, Relative Distribution
    """
    os.makedirs(f'./{mg_output}', exist_ok = True)
    rf[name_var]=pd.Categorical(rf[name_var])
    point=rf['periodo'].max().__str__()[:10]
    if load_var in cols['cat_features']:
        agg=rf[name_var].value_counts().index[0]
        plot_cat_timeseries(rf,load_var,name_var,name_fields,agg,point,mg_output)
        plot_cat_relative(rf,load_var,name_var,name_fields,mg_output,pal)
        plot_cat_absolute(rf,load_var,name_var,name_fields,mg_output,pal)
    elif load_var in cols['cont_features']:
        plot_num_timeseries(rf,load_var,name_var,name_fields,'mean',point,mg_output)
        plot_num_relative(rf,load_var,name_var,name_fields,mg_output,pal)
        plot_num_absolute(rf,load_var,name_var,name_fields,mg_output,pal)
def plot_correlation(df,mg_output):
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize = (14, 14))
    plt.title('Correlation matrix', size = 23)
    rf=df.corr()
    sns.heatmap(rf,mask = np.triu(rf),linecolor='black',vmin = -1, vmax = 1,cmap='coolwarm_r',cbar=False)
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    plt.savefig(f'{mg_output}/correlation_matrix.png',dpi=150, bbox_inches='tight')
    return fig
def st_bars(df,f,target,sub_cols):
    n_classes=len(sub_cols)
    df.dropna(subset=[f],inplace=True)
    t_df=pd.crosstab(df[target],df[f],normalize='columns')
    lst=[]
    cols_s=[]  
    for i in range(n_classes):
        lst.extend(t_df.iloc[i,:].tolist())
        cols_s=cols_s+[sub_cols[i]]*len(t_df.columns.tolist())
    ts_df=pd.DataFrame({
            target:cols_s
        ,f: t_df.columns.tolist()*n_classes,'group_feature':t_df.columns.tolist()*len(sub_cols),'WG': lst
      })
    ts_df[f]=ts_df[f].apply(lambda x: str(x))
    
    return ts_df,ts_df[f].unique()
def timeseries_get_bands(column,window_percentage = 0.2):
    N = len(column)
    time = column.reset_index()['periodo']
    k = int(len(column) * (window_percentage))
    N = len(column)    
    column.reset_index(inplace=True,drop=True)
    get_bands = lambda data : (np.mean(data) + 2.5*np.std(data),np.mean(data) - 2.5*np.std(data))
    bands = [get_bands(column[range(0 if i - k < 0 else i-k ,i + k if i + k < N else N)]) for i in range(0,N)]
    upper, lower = zip(*bands)
    anomalies = (column > upper) | (column < lower)
    return time,column,anomalies,upper, lower
def plot_timeseries(mf,f,target,sub_cols,point,mg_output):
    l=mf.reset_index()[target].unique()
    mf=mf.reset_index().set_index(target)
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(14,7))
    for n,x in enumerate(l): 
        # mf.loc[x][f].plot(color='gray',style='o-',grid=True,label=f'Timeseries_{sub_cols[x]}', use_index = True)
        plt.plot(mf.loc[x]['periodo'],mf.loc[x][f],'o-',label=f'Timeseries_{sub_cols[n]}',color=colors[n])
        time,column,anomalies,upper,lower=timeseries_get_bands(mf.loc[x].set_index('periodo')[f])
        rf=pd.DataFrame({'periodo':time,'upper':upper,'lower':lower,'anomalies':anomalies,'columns':column})
        if n==0:
            plt.plot(rf[rf['anomalies']]['periodo'],rf[rf['anomalies']]['columns'],'ro',label='Anomalies')
            plt.plot(rf['periodo'],rf['upper'],'r-',label='Bands',alpha=0.1)
        else:
            plt.plot(rf[rf['anomalies']]['periodo'],rf[rf['anomalies']]['columns'],'ro')
            plt.plot(rf['periodo'],rf['upper'],'r-',alpha=0.1)
        plt.plot(rf['periodo'],rf['lower'],'r-',alpha=0.1)
        plt.fill_between(time, upper, lower,facecolor='red',alpha=0.1)
    plt.axvline(pd.to_datetime(point),linestyle='dashed',color='blue',label='Actual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.title(f)
    plt.legend(loc='lower center',frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102),
           mode='expand',
           ncol=4,
           borderaxespad=-.46,
           prop={'size': 10, 'family':'Calibri'})
    plt.savefig(f'{mg_output}/{f}_{sub_cols[-1]}_timeseries.png',dpi=150, bbox_inches='tight')
    return fig
def plot_cat_relative(dfx,f,target,sub_cols,mg_output,pal='mako'):
    if len(dfx[f].value_counts())>10: df=dfx[dfx[f].isin(dfx[f].value_counts().index.tolist()[:10])]
    else: df=dfx.copy()
    df[f]=df[f].astype(str)
    xf,ixf=st_bars(df,f,target,sub_cols)
    df[f]=pd.Categorical(df[f],ixf)
    fig,ax=plt.subplots(nrows=1,ncols=1)
    sns.histplot(data=xf,y=f,hue=target,multiple='stack',weights='WG',palette=pal,shrink=0.9)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x+width/2,
                y+height/2,
                '{:.3f}'.format(width),
                horizontalalignment='center',
                verticalalignment='center',
                color='black',
                fontsize=10,
                **{'fontname':'DejaVu Sans'})
    ax.legend(sub_cols,loc='center',
           frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102),
           mode='expand',
           ncol=4,
           borderaxespad=-.46,
           prop={'size': 10, 'family':'Calibri'})
    plt.savefig(f'{mg_output}/{f}_{sub_cols[-1]}_relative.png',dpi=150, bbox_inches='tight')
    return fig
def plot_cat_absolute(dfx,f,target,sub_cols,mg_output,pal='mako'):
    if len(dfx[f].value_counts())>11: df=dfx[dfx[f].isin(dfx[f].value_counts().index.tolist()[:11])]
    else: df=dfx.copy()
    df[f]=pd.Categorical(df[f])
    fig,ax=plt.subplots(nrows=1,ncols=1)
    sns.countplot(data=df,y=f, hue=target, palette=pal)
    plt.legend(sub_cols,loc='center',
           frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102),
           mode='expand',
           ncol=4,
           borderaxespad=-.46,
           prop={'size': 10, 'family':'Calibri'})
    plt.savefig(f'{mg_output}/{f}_{sub_cols[-1]}_absolute.png',dpi=150, bbox_inches='tight') 
    return fig
def plot_num_relative(df,f,target,sub_cols,mg_output,pal='mako'):
    fig,ax=plt.subplots(nrows=1,ncols=1)
    sns.violinplot(data=df, y=target,x=f,scale='count', palette=pal)
    plt.legend(sub_cols,loc='center',
           frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102),
           mode='expand',
           ncol=4,
           borderaxespad=-.46,
           prop={'size': 10, 'family':'Calibri'})
    plt.savefig(f'{mg_output}/{f}_{sub_cols[-1]}_relative.png',dpi=150, bbox_inches='tight')   
    return fig
def plot_num_absolute(df,f,target,sub_cols,mg_output,pal='mako'):
    fig,ax=plt.subplots(nrows=1,ncols=1)
    sns.kdeplot(data=df,x=f, hue=target ,fill=True, palette=pal)
    plt.legend(sub_cols,loc='center',
           frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102),
           mode='expand',
           ncol=4,
           borderaxespad=-.46,
           prop={'size': 10, 'family':'Calibri'})
    plt.savefig(f'{mg_output}/{f}_{sub_cols[-1]}_absolute.png',dpi=150, bbox_inches='tight')
    return fig
def plot_cat_timeseries(dfx,f,target,sub_cols,agg,point,mg_output):
    dfx['periodo']=pd.to_datetime(dfx['periodo'])
    if dfx[f].nunique()>10: df=dfx[dfx[f].isin(dfx[f].unique()[:10])]
    else: df=dfx.copy()
    mf=df[df[f]==agg].groupby([target,'periodo'])[[f]].agg('count')
    return plot_timeseries(mf,f,target,sub_cols,point,mg_output)
def plot_num_timeseries(df,f,target,sub_cols,agg,point,mg_output):
    df['periodo']=pd.to_datetime(df['periodo'])
    mf=df.groupby([target,'periodo'])[[f]].agg(agg)
    return plot_timeseries(mf,f,target,sub_cols,point,mg_output)
def get_pie_value(dfx,v,f,agg='sum'):
    m=dfx.groupby(f)[v].agg(agg).reset_index()
    m=m.sort_values(v,ascending=False)[:12]
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=m[f]
                         , values=m[v[0]].round(3)
                         , name=v[0]),
              1, 1)
    fig.add_trace(go.Pie(labels=m[f]
                                      , values=m[v[1]].round(3), name=v[1]),
              1, 2)
    
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=10,
                      marker=dict(colors=['rgb(255, 102, 102)','rgb(102, 178, 255)'], line=dict(color='#000000', width=2)))
    fig.update_layout(title_text=f'{v[0]} y {v[1]} de {f}'    ,legend=dict(orientation='h')
)
    return fig