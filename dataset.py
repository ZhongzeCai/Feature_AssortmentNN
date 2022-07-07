import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def GrepDataset(
    data_seed=1300,
    data_path="data/xWy_10_3ProdF_5CusF",
    name="xWy_10_3ProdF_5CusF",
    train_amount=2000,
    valid_amount=200,
    test_amount=2000,
    device="cpu",
    **kwargs
):

    random.seed(data_seed)

    IN = torch.Tensor(np.load(data_path+"/"+name+"_IN.npy"))
    SAMP = torch.Tensor(np.load(data_path+"/"+name+"_SAMP.npy"))
    PROB = torch.Tensor(np.load(data_path+"/"+name+"_PROB.npy"))

    if device == "gpu":
        IN = IN.to('cuda')
        SAMP = SAMP.to('cuda')
        PROB = PROB.to('cuda')


    total_data = len(IN)
    total_amount = train_amount + valid_amount + test_amount

    positions = random.sample(list(range(total_data)),k=total_amount)

    training_positions = positions[:train_amount]
    validating_positions = positions[train_amount:train_amount+valid_amount]
    testing_positions = positions[train_amount+valid_amount:total_amount]

    training_dataset = TensorDataset(
        IN[training_positions],
        SAMP[training_positions],
        PROB[training_positions]
    )
    
    validating_dataset = TensorDataset(
        IN[validating_positions],
        SAMP[validating_positions],
        PROB[validating_positions]
    )

    testing_dataset = TensorDataset(
        IN[testing_positions],
        SAMP[testing_positions],
        PROB[testing_positions]
    )
    
    

    return training_dataset, validating_dataset, testing_dataset


if __name__ == "__main__":
    training_dataset, validating_dataset, testing_dataset = GrepDataset()
    print(training_dataset[0][0].numpy())
    print(training_dataset[0][1].numpy())
    print(training_dataset[0][2].numpy())
