param = {}
param['max_depth']= 4   # depth of tree
param['eta'] = 0.18      # shrinkage parameter
param['verbosity'] = 0  # 0= no logging 3=max logging
param['objective'] = 'binary:logistic'
param['nthread'] = 7 # Number of threads used
param['eval_metric'] = ['error','logloss']
param['feature_selector']='shuffle'
param['early_stopping_rounds']=14


def param_D2L(param):
    """Translate a param dictionary to a param list"""
    plist=param.items()
    new_plist=[]
    for p in plist:
        if type(p[1])!=list:
            new_plist.append(p)
        else:
            for e in p[1]:
                new_plist.append((p[0],e))
    return new_plist
