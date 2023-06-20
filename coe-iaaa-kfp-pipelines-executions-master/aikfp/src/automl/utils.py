def tryconvert(x,d,f):
    try: return f(x)
    except: return d
def exist_element_cfg(cfg,x,n):
    try: cfg[x]
    except: cfg[x]=n    
