import numpy as np
from .AssortmentGenerator import Product_0, Product_1
from .BaseGenerator import BaseGenerator
import os


def Pvec_to_Choice(p_vec):
    if not sum(p_vec) > 1.0 - 1e-6 and sum(p_vec) < 1.0 + 1e-6:
        print("WRONG PROBABILITY!")
        return
    index = np.random.choice(len(p_vec), 1, p=p_vec)[0]
    indicate = np.zeros(len(p_vec))
    indicate[index] = 1
    return indicate


class GenTrueModel_Markov(BaseGenerator):

    def __init__(

        self, 
        N_prod = 10, 

        Len_prodFeature = 3,
        Len_customerFeature = 5,

        W_scale = 1,

        prodFeature_genFunc = lambda n : np.random.normal(0,1,n),

        **kwargs

    ):

        super().__init__(N_prod, Len_prodFeature, Len_customerFeature)

        self.ModelName = "Markov"

        

        self.W_scale = W_scale

        self.W_markov = np.random.normal(loc=0, scale=W_scale, size=(Len_prodFeature + Len_customerFeature, self.N_prod + 1))
        self.b_markov = np.random.normal(loc = 0, scale=W_scale, size=self.N_prod + 1)
        self.W_lambda = np.random.normal(loc=0, scale=W_scale, size=(Len_customerFeature, self.N_prod))
        self.b_lambda = np.random.normal(loc=0, scale=W_scale, size=self.N_prod)


        self.prodFeature_genFunc = prodFeature_genFunc
        
        self.products = np.zeros((self.N_prod, self.Len_prodFeature))
        self.self_gen_instance()


    def self_gen_instance(self):
        # generate all products

        for row in range(self.N_prod):
            
            self.products[row] = self.prodFeature_genFunc(self.Len_prodFeature)

    def markov_para_gen(self, customer_feature):

        Lams = np.exp(np.matmul(customer_feature, self.W_lambda) + self.b_lambda)
        Lams = Product_0(Lams / sum(Lams))
        TransP_root  = np.zeros((self.N_prod + 1, self.Len_prodFeature + self.Len_customerFeature))

        TransP_root[0] = np.append(
            np.array([0]*self.Len_prodFeature),
            customer_feature
        )
        for i in range(1, self.N_prod + 1):
            TransP_root[i] = np.append(
                self.products[i-1],
                customer_feature
            )

        TransP = np.matmul(TransP_root, self.W_markov)
        for i in range(0, self.N_prod + 1):
            pros = np.exp(TransP[i] + self.b_markov)
            pros = pros / sum(pros)
            TransP[i] = pros

        return (Lams, TransP)


    def prob_for_assortment(self, prod_assort, customer_feature):

        Lams, TransP = self.markov_para_gen(customer_feature)

        Assorts = Product_1(prod_assort)

        S_plus = np.squeeze(np.argwhere(Assorts == 1),axis=1)
        S_bar = np.squeeze(np.argwhere(Assorts == 0),axis=1)
        B = TransP[np.expand_dims(S_bar, axis=1), S_plus]
        C = TransP[np.expand_dims(S_bar, axis=1), S_bar]
    
        distri = np.zeros(len(Lams))
    
        addi = np.matmul(np.matmul(np.expand_dims(Lams[S_bar], axis=0), np.linalg.inv(np.identity(len(C)) - C)), B)
    
        count = 0
        for i in S_plus:
            distri[i] = Lams[i] + addi[0,count]
            count += 1
    
        return distri

    # the output is also one_hot encoded
    def gen_final_choice(self, prod_assort, customer_feature):

        distri = self.prob_for_assortment(prod_assort, customer_feature)

        return Pvec_to_Choice(distri)


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
            

        np.save(folder_path+"/products", self.products)

        np.save(folder_path+"/W_markov", self.W_markov)
        np.save(folder_path+"/W_lambda", self.W_lambda)
        np.save(folder_path+"/b_markov", self.b_markov)
        np.save(folder_path+"/b_lambda", self.b_lambda)


if __name__ == "__main__":

    model = GenTrueModel_Markov(
        N_prod = 5, 

        Len_prodFeature = 3,
        Len_customerFeature = 5,
    )

    print(model.products)
    assortment = np.array([1,0,1,1,0])
    customer_feature = np.array([1,1,0,1,0])

    Lams, TransP = model.markov_para_gen(customer_feature)
    print(Lams)
    print(np.sum(TransP, axis = 1))





