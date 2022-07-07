from experiment import Experiment
import yaml
from models import ModelCollection
from dataset import GrepDataset
from torch.utils.data import DataLoader
import numpy as np


# run in jupyter notebook!
def runner(yaml_path):
    with open(yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    data_path = params['data_params']['data_path']

    prods = np.load(data_path+'/products.npy')

    model = ModelCollection[params['model_params']['name']](**params['model_params'], products = prods)

    training_dataset, validating_dataset, testing_dataset = GrepDataset(**params['data_params'])

    T_loader = DataLoader(
        training_dataset, shuffle=True, batch_size = params['exp_params']['train_batch_size']
    )

    V_loader = DataLoader(
        validating_dataset, shuffle=False, batch_size = params['exp_params']['valid_batch_size']
    )

    Te_loader = DataLoader(
        testing_dataset, shuffle=False, batch_size = len(testing_dataset)
    )


    EXP = Experiment(
        model,
        T_loader,
        V_loader,
        Te_loader,
        params['exp_params'],
        params['logging_params']
    )

    EXP.run(echo=False)

