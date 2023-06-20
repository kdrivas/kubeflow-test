import logging
import psutil
import pandas as pd
import numpy as np
from . import utils
def save_system_logs(name):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(filename=f'{name}.log',level=logging.INFO,
format=f"""%(asctime)s 
%(relativeCreated)d 
%(name)s 
%(levelname)s 
%(message)s""")
    return logging
def save_system_line(logging,message):
    logging.info(f"""
{dict(psutil.virtual_memory()._asdict())}
{dict(psutil.cpu_times_percent()._asdict())}
{[x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]}
{message}\t\t""")
def calculate_costs(name):
    with open(f'{name}.log') as f: m=f.read()
    rf=pd.DataFrame([[y for y in x.split('\n') if y!=''][:8] for x in m.split('\t\t')],columns=['periodo','time_acum','modulename','levelname','memory_usage','cpu_times_percent','cpu_usage','message']).iloc[:-1]
    rf['periodo']=pd.to_datetime(rf['periodo'])
    def ws(x):
            return dict((str(a.strip().replace('{','').replace("'",''))
                         ,float(b.strip().replace('}','').replace("'",'')))  
                         for a, b in (element.split(':')  
                                      for element in x.split(', ')[:-1]
                                     ))           
    def wr(x,y):
        try: return x[y]
        except: return np.NAN
    rf['memory_usage']=rf['memory_usage'].apply(ws)
    rf['cpu_times_percent']=rf['cpu_times_percent'].apply(ws)
    for y in ['total', 'available', 'percent', 'used', 'free']: 
        rf[f'memory_usage_{y}']=rf['memory_usage'].apply(lambda x:wr(x,y))
    for y in ['user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'guest', 'guest_nice']:
        rf[f'cpu_times_percent_{y}']=rf['cpu_times_percent'].apply(lambda x:wr(x,y))
    # for n,y in enumerate(['-1','0','+1']): rf[f'cpu_usage_{y}']=rf['cpu_usage'].apply(lambda x:utils.tryconvert(x,np.NAN, lambda w: w.replace('[','').replace(']','')[n])
    for n,y in enumerate(['m_1','0','s_1']): rf[f'cpu_usage_{y}']=rf['cpu_usage'].apply(lambda x:utils.tryconvert(x,np.NAN, lambda w: eval(w)[n]))
    del rf['cpu_times_percent'],rf['memory_usage'],rf['cpu_usage']
    return rf