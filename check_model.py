## a series of functions for testing the model
from experiment import KL_loss, Relative_L1_loss
import yaml
from models import ModelCollection
from dataset import GrepDataset
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def silly_model(IN):

    OUT = torch.randn_like(IN)

    OUT[:0]=0
    
    probs = nn.Softmax(dim=-1)(OUT)
        
        
    OUT = IN * probs

    OUT = OUT / OUT.sum(dim=-1).unsqueeze(-1)

    return OUT


def Loss_check(log_path = 'logs/xWy_10_xWyModel'):

    train_loss = np.load(log_path+"/log_train_loss.npy")
    valid_loss = np.load(log_path+"/log_valid_loss.npy")

    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def model_behavior_check(training_yaml_path, model_name, model_path=""):

    with open(training_yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    if model_path == "":
        model_path = "logs/"+model_name+"/"+model_name+"_last.pth"

    training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])

    model = torch.load(model_path)

    IN, SAMP, PROB = testing_dataset[:]

    print("input :")
    print(IN[0:5])

    print("actual probs:")
    print(PROB[0:5])

    print("predicted probs:")
    OUT = model(IN)
    print(OUT[0:5])

    print("KL loss:")
    print(KL_loss(OUT, PROB))

    print("min KL loss:")
    print(KL_loss(PROB, PROB))

    print("silly model KL loss: ")
    IN = IN[:, :params['model_params']['Vec_Len']]
    Silly_OUT = silly_model(IN)
    print(KL_loss(Silly_OUT, PROB))

def L1_loss_check(training_yaml_path, model_name, model_path = "", clamp_level=10):

    with open(training_yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    if model_path == "":
        model_path = "logs/"+model_name+"/"+model_name+"_last.pth"

    training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])

    model = torch.load(model_path)

    (IN, SAMP, PROB) = testing_dataset[:]

    relative_diff = Relative_L1_loss(model(IN), PROB, clamp_level)

    plt.hist(relative_diff)

    plt.show()


    