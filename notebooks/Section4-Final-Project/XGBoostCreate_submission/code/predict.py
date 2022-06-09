from lib.KDTreeEncoding import *

import xgboost as xgb
from lib.XGBHelper import *
from lib.XGBoost_params import *
from lib.score_analysis import *

from lib.logger import logger

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from glob import glob
import pandas as pd
import pickle as pkl
import sys
from time import time
class timer:
    def __init__(self):
        self.t0=time()
        self.ts=[]
    def mark(self,message):
        self.ts.append((time()-self.t0,message))
        print('%6.2f %s'%self.ts[-1])
    def _print(self):
        for i in range(len(self.ts)):
            print('%6.2f %s'%self.ts[i])


T=timer()
            
poverty_dir=sys.argv[1]
image_dir=poverty_dir+'anon_images/'
depth=8   #for KDTree

import pickle as pkl

pickle_file_urban='data/checkpoint_Urban.pk'
pickle_file_rural='data/checkpoint_Rural.pk'

D_urban=pkl.load(open(pickle_file_urban,'rb'))
D_rural=pkl.load(open(pickle_file_rural,'rb'))

d_mean_urban = D_urban['mean']
d_std_urban=D_urban['std']

d_mean_rural = D_rural['mean']
d_std_rural=D_rural['std']


#read out the ensemble of classifiers.
bst_list_urban=[x['bst'] for x in D_urban['styled_logs'][1]['log']]
bst_list_rural=[x['bst'] for x in D_rural['styled_logs'][1]['log']]

T.mark('read pickle files')

# ## Iterate over test sets
folds=[{'in':'country_test_reduct.csv','out':'results_country.csv'},
      {'in':'random_test_reduct.csv','out':'results.csv'}]

for fold_i in range(len(folds)):
    fold=folds[fold_i]

    #load table entries
    test_csv=f'../public_tables/{fold["in"]}'
    test=pd.read_csv(test_csv,index_col=0)
    test.index=test['filename']
    test.shape

    out=pd.DataFrame()
    out['filename'] = test['filename']
    out['urban']=test['urban']
    out['pred_wo_abstention']=0
    out.set_index('filename', inplace=True)
    selector = out['urban']

    ## Encode all data using encoding tree
    Enc_data=encoded_dataset(image_dir,out,D_urban['tree'],label_col='pred_wo_abstention')

    data=to_DMatrix(Enc_data.data)
    Preds = zeros([Enc_data.data.shape[0],len(bst_list_urban)])
    
    for i in range(Enc_data.data.shape[0]):
        if selector[i]:
            for i in range(len(bst_list_urban)):
                Preds[:,i]=bst_list_urban[i].predict(data,output_margin=True)
        else:
            for i in range(len(bst_list_rural)):
                Preds[:,i]=bst_list_rural[i].predict(data,output_margin=True)
    
    Preds[selector]=(Preds[selector]-d_mean_urban)/d_std_urban # apply overall score scaling
    Preds[~selector]=(Preds[~selector]-d_mean_rural)/d_std_rural

    _mean=np.mean(Preds,axis=1)
    _std=np.std(Preds,axis=1)

    pred_wo_abstention=(2*(_mean>0))-1
    pred_with_abstention=copy(pred_wo_abstention)
    pred_with_abstention[_std>abs(_mean)]=0
    
    out['pred_with_abstention'] = pred_with_abstention
    out['pred_wo_abstention'] = pred_wo_abstention
  
    outFile=f'data/{fold["out"]}'
    out.to_csv(outFile)
    print('\n\n'+'-'*60)
    print(outFile)
    T.mark('generated '+outFile)

