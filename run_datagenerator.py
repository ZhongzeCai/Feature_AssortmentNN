from datagenerator import GenDataset
import numpy as np


all_configs = [
    'xWy_10_3ProdF_5CusF',
    'xWy_10_3ProdF_5CusF_Scale03',
    'xWy_10_3ProdF_5CusF_Scale5'
]

for config in all_configs:
    path = "gen_config/"+config+".yaml"
    GenDataset(config_path=path)

    '''
    products = np.load('data/'+config+'/products.npy')

    print(products)

    IN = np.load('data/'+config+'/'+config+'_IN.npy')

    print(IN[0:5])
    '''