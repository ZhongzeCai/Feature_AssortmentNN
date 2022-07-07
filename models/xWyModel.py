import numpy as np
import torch
import torch.nn as nn


class xWyModel(nn.Module):

    def __init__(self, 
        Vec_Len, 
        model_seed=1234, 
        
        Len_prodFeature = 3,
        Len_customerFeature = 5,
        products = np.array([]),
        
        **kwargs
    ):


        super(xWyModel, self).__init__()

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

        self.W = torch.nn.Parameter(
            torch.randn((self.Len_prodFeature, self.Len_customerFeature),requires_grad=True)
        )
        

        self.softmax = nn.Softmax(dim=-1)


    def misc_to_gpu(self):
        self.products = self.products.cuda()

    def forward(self, IN, **kwargs):

        assorts = IN[:, :self.Vec_Len]
        cusFs = IN[:, self.Vec_Len:]

        # utilities
        cusFs = torch.unsqueeze(cusFs, -1)
        
        util_mat = torch.squeeze(torch.matmul(torch.matmul(self.products, self.W), cusFs), dim=-1)

        probs = self.softmax(util_mat)
        
        
        x = assorts * probs

        x = x / x.sum(dim=-1).unsqueeze(-1)

        return x


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

    model = xWyModel(
        Vec_Len,
        model_seed, 
            
        Len_prodFeature,
        Len_customerFeature,
        products
    )

    print(model.W)
    print(model.products)
    data_batch = torch.Tensor([
        [1,0,0,1,0, 0.1, 0.4, 0.3],
        [1,1,1,1,1, 0.5, 0.7, 0.2],
        [1,0,1,1,1, 0.6, 0.1, 0.1],
        [1,0,1,0,1, 0.9, 0.1, 0.1],
        [1,0,1,0,0, 0.6, 0.8, 0.1],
        [1,0,0,0,1, 0.6, 0.1, 0.3]
    ])

    print(model(data_batch))

