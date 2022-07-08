import os
import yaml
import generator
from generator import AssortmentGenerator
from generator.AssortmentGenerator import Product_0, Product_1
import generator
import numpy as np
import random


def GenDataset(config_path = "gen_config/xWy_10_3ProdF_5CusF.yaml"):

    with open(config_path, 'r') as file:
        params = yaml.safe_load(file)


    seed = params['Seed']
    np.random.seed(seed)
    random.seed(seed)

    name = params['DataSet']['name']
    N_sample = params['DataSet']['N_sample']

    AssortParams = params['Assortment']
    TrueModelParams = params['TrueModel']

    path = 'data/' + name
    if not os.path.exists(path):
        os.makedirs(path)

    dir = os.listdir(path)
    if len(dir) > 0:
        print("Non empty directory")
        return

    with open(path+"/Dataset_description.txt", 'w') as f:
        f.write("dataset name : "+name)
        f.write("\n")
        f.write("sample amount : "+str(N_sample))
        f.write("\n")
        f.write("config file : "+config_path)


    AssortmentGen = AssortmentGenerator.GenAssortment[AssortParams['scheme']]

    TrueModel = generator.TrueModel[TrueModelParams['model']](**TrueModelParams)

    TrueModel.save_para(path)

    N_prod = TrueModelParams['N_prod']
    Vec_Len = N_prod + 1

    Len_customerFeature = TrueModelParams['Len_customerFeature']

    INPUT = np.zeros((N_sample,Vec_Len + Len_customerFeature))
    SAMP_OUTPUT = np.zeros((N_sample,Vec_Len))
    PROB_OUTPUT = np.zeros((N_sample,Vec_Len))

    for i in range(N_sample):
        
        assort = AssortmentGen(**AssortParams)

        # this part can be changed to other generating function
        customer_feature = np.random.normal(0, 1, Len_customerFeature)

        INPUT[i] = np.append(Product_1(assort), customer_feature) 

        SAMP_OUTPUT[i] = TrueModel.gen_final_choice(assort, customer_feature)
        
        PROB_OUTPUT[i] = TrueModel.prob_for_assortment(assort, customer_feature)


    np.save(path+"/"+name+"_IN",INPUT)
    np.save(path+"/"+name+"_SAMP",SAMP_OUTPUT)
    np.save(path+"/"+name+"_PROB",PROB_OUTPUT)

    print("dataset "+name + " generated!")



