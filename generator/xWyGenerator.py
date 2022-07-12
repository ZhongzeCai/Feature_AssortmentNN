## we assume a limited number of possible products, but an unlimited number of customers

import numpy as np
from .AssortmentGenerator import Product_0, Product_1
from .BaseGenerator import BaseGenerator
import os


class GenTrueModel_xWy(BaseGenerator):

    def __init__(

        self, 
        N_prod = 10, 

        Len_prodFeature = 3,
        Len_customerFeature = 5,

        W_scale = 1,

        W = np.array([]),

        prodFeature_genFunc = lambda n : np.random.normal(0,1,n),

        **kwargs

    ):

        super().__init__(N_prod, Len_prodFeature, Len_customerFeature)

        self.ModelName = "xWy"

        if len(W) == 0:

            self.W_scale = W_scale

            W = np.random.normal(loc=0, scale=W_scale, size=(Len_prodFeature, Len_customerFeature))

        self.W = W

        self.prodFeature_genFunc = prodFeature_genFunc
        
        self.products = np.zeros((self.N_prod, self.Len_prodFeature))
        self.self_gen_instance()


    def self_gen_instance(self):
        # generate all products

        for row in range(self.N_prod):
            
            self.products[row] = self.prodFeature_genFunc(self.Len_prodFeature)


    # assume that assortment is already one-hot encoded
    # and does not include product 0
    def prob_for_assortment(self, prod_assort, customer_feature):

        utils = np.squeeze(np.matmul( self.products , np.matmul(self.W, customer_feature.T) ))

        utils = Product_0(utils)

        prod_assort = Product_1(prod_assort)

        probs = np.exp(utils) * prod_assort

        probs = probs / sum(probs)

        return probs

    # the output is also one_hot encoded
    def gen_final_choice(self, prod_assort, customer_feature):

        fin = np.zeros(self.N_prod + 1)
        
        probs = self.prob_for_assortment(prod_assort, customer_feature)

        choice = np.random.choice(self.N_prod + 1, 1, p=probs)[0]

        fin[choice] = 1

        return fin


    def save_para(self, folder_path):

        with open(folder_path+"/TrueModel_description.txt", 'w') as f:
            f.write("model type : "+self.ModelName)
            f.write("\n")
            f.write("product number : "+str(self.N_prod))
            f.write("\n")
            f.write("product feature length : "+str(self.Len_prodFeature))
            f.write("\n")
            f.write("customer feature length : "+str(self.Len_customerFeature))
            f.write("\n")
            

        np.save(folder_path+"/W", self.W)
        np.save(folder_path+"/products", self.products)




if __name__ == "__main__":

    xWy = GenTrueModel_xWy(
        N_prod = 5, 

        Len_prodFeature = 3,
        Len_customerFeature = 5,

        W = np.array([]),

        prodFeature_genFunc = lambda n : np.random.normal(0,1,n),
    )

    print(xWy.W)
    print(xWy.products)
    assortment = np.array([1,0,1,1,0])
    customer_feature = np.array([1,1,0,1,0])

    print(xWy.prob_for_assortment(assortment, customer_feature))
    print(xWy.gen_final_choice(assortment, customer_feature))




