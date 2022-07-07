import numpy as np
import random

# for given 0 < lam < 1, given N candidate products, generated assortment will contain averagely lam * N products
def GenAssortment_Even(N_prod = 10, lam=1/2, **kwargs):
    potential_vec = np.random.uniform(low=0., high=1, size=N_prod)
    assortment_vec = np.zeros(N_prod)
    assortment_vec[potential_vec <= lam] = 1
    return assortment_vec

def GenAssortment_Sparse(N_prod = 10, sparse_fun = lambda x : np.sqrt(x)/x, **kwargs):
    return GenAssortment_Even(N_prod, sparse_fun(N_prod))

# generate assortment containing fixed number of products
def GenAssortment_Fixed(N_prod = 10, fixed_num = 6, **kwargs):
    positions = random.sample(list(range(N_prod)),k=fixed_num)
    assortment_vec = np.zeros(N_prod)
    assortment_vec[positions] = 1
    return assortment_vec

def GenAssortment_Abundant(N_prod = 10, **kwargs):
    fixied = random.sample(list(range(1,N_prod+1)),k=1)[0]
    return GenAssortment_Fixed(N_prod, fixied)

# in some cases we want product 0 in assortment, so use the following function
def Product_0(prod_assort):
    return np.insert(prod_assort, 0, 0)

# sometimes we want the encoding of product 0 to be 1
def Product_1(prod_assort):
    return np.insert(prod_assort, 0, 1)



GenAssortment = {
    "Even" : GenAssortment_Even,
    "Sparse": GenAssortment_Sparse,
    "Fixed": GenAssortment_Fixed,
    "Abundant": GenAssortment_Abundant
}


if __name__ == "__main__":
    print(GenAssortment_Even(N_prod = 10, lam = 1/3))
    print(GenAssortment_Sparse(N_prod = 10))
    print(GenAssortment_Fixed(N_prod=10, fixed_num=3))
    print(GenAssortment_Abundant(10))
