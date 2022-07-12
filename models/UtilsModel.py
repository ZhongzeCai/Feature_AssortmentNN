import numpy as np
import torch
import torch.nn as nn


class UtilsModel(nn.Module):

    def __init__(self, 
        Vec_Len, 
        model_seed=1234, 
        
        Len_prodFeature = 3,
        Len_customerFeature = 5,
        products = np.array([]),

        Num_cusEncoder_midLayer = 2,
        cusEncoder_midLayers = [3, 3],

        Num_prodEncoder_midLayer = 2,
        prodEncoder_midLayers = [3, 3],

        Num_CrossEffectLayer = 1,
        CrossEffectLayers = [5],
        
        **kwargs
    ):


        super(UtilsModel, self).__init__()

        self.Vec_Len = Vec_Len

        torch.manual_seed(model_seed)

        self.Len_prodFeature = Len_prodFeature
        self.Len_customerFeature = Len_customerFeature

        prod_0 = np.zeros((1, self.Len_prodFeature))
        self.products = torch.Tensor(
            np.concatenate((prod_0, products), axis=0)
        )

        if(self.Vec_Len != len(self.products)):
    
            raise Exception("product amount not match!")

        self.Num_cusEncoder_midLayer = Num_cusEncoder_midLayer
        self.cusEncoder_midLayers = cusEncoder_midLayers

        self.Num_prodEncoder_midLayer = Num_prodEncoder_midLayer
        self.prodEncoder_midLayers = prodEncoder_midLayers


        self.Num_CrossEffectLayer = Num_CrossEffectLayer
        self.CrossEffectLayers = CrossEffectLayers


        if(self.cusEncoder_midLayers[-1] != self.prodEncoder_midLayers[-1]):
    
            raise Exception("cross utility layer not match!")


        ## cusEncoder network
        self.cusEncoder = self.gen_sequential(self.Len_customerFeature, self.cusEncoder_midLayers)

    
        ## prodEncoder network
        self.prodEncoder = self.gen_sequential(self.Len_prodFeature, self.prodEncoder_midLayers)

        ## res layer
        self.Res = nn.Linear(2, 1)

        ## activate
        self.sigmoid = nn.Sigmoid()

        ## CrossEffect network
        self.CrossEffect = self.gen_sequential(self.Vec_Len, np.append(self.CrossEffectLayers, self.Vec_Len))

        ## normalize
        self.softmax = nn.Softmax(dim=-1)



    def gen_sequential(self, f_channel, midLayers):

        modulist = []
        from_channel = f_channel
        for layer in range(len(midLayers)):

            to_channel = midLayers[layer]

            sub_net = nn.Sequential(
                nn.Linear(from_channel, to_channel),
                nn.BatchNorm1d(to_channel),
                nn.LeakyReLU(0.2, inplace=True)
            )
            modulist.append(sub_net)

            from_channel = to_channel

        return nn.ModuleList(modulist)



    def misc_to_gpu(self):
        self.products = self.products.cuda()

    def forward(self, IN, **kwargs):

        assorts = IN[:, :self.Vec_Len]
        cusFs = IN[:, self.Vec_Len:]

        ## encode products
        prod_features = self.products
        for m in self.prodEncoder:
            prod_features = m(prod_features)
    

        ## encode customers
        cus_features = cusFs
        for m in self.cusEncoder:
            cus_features = m(cus_features)


        ## utility
        cus_features = torch.unsqueeze(cus_features, 1)

        encoded_utils = torch.matmul(cus_features, prod_features.T)

        residual = torch.unsqueeze(assorts, 1)

        encoded_utils = torch.cat((encoded_utils, residual), 1).permute(0, 2, 1)

        encoded_utils = self.sigmoid(torch.squeeze(self.Res(encoded_utils),2))

        ## cross effect
        prob = encoded_utils
        for m in self.CrossEffect:
            prob = m(prob)

        prob = self.softmax(prob)

        ## regularize

        prob = prob * assorts
        prob = prob / prob.sum(dim=-1).unsqueeze(-1)

        return prob



if __name__ == "__main__":

    Vec_Len = 5
    model_seed=1234 
        
    Len_prodFeature = 2
    Len_customerFeature = 3
    products = np.array([
        [1, 0],
        [0, 1],
        [0.5, 0.5],
        [0.3, 0.2]
    ])

    model = UtilsModel(
        Vec_Len,
        model_seed, 
            
        Len_prodFeature,
        Len_customerFeature,
        products
    )

    print(model.products)
    data_batch = torch.Tensor([
        [1,0,0,1,0, 0.1, 0.4, 0.3],
        [1,1,1,1,1, 0.5, 0.7, 0.2],
        [1,0,1,1,1, 0.6, 0.1, 0.1],
        [1,0,1,0,1, 0.9, 0.1, 0.1],
        [1,0,1,0,0, 0.6, 0.8, 0.1],
        [1,0,0,0,1, 0.6, 0.1, 0.3]
    ])

    ## print(model(data_batch))
    print("---------------debug-------------------")
    model(data_batch)

