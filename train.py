import argparse
import numpy as nps
from utils.data import Data
import models

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def train(data, config):
    
    
    # build model and if reload checkpoint
    model = getattr(models, config.model)(config)
    if data.checkpoints:
        model.load_state_dict(data.checkpoints['model'])
    model = model.to(config.device)
    
    # optimizer
    if data.checkpoints:
        optim = checkpoints['optim']
    else:
        optim = None # Fix
    
    for idx in range(config.epoch):
        model.train()

        for batch in data.trainloader:
            model.zero_grad()

        src, src_len, tgt, tgt_len = batch['src'], batch['src_len'], batch['tgt'], batch['tgt_len']

        
def main():
    # config
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-config', default='default.yaml', type=str,
                        help='config file')

    opt = parser.parse_args()


    config = read_config(opt.config)
    # prepare data
    data = Data(config)
    config = data.config  # if reload checkpoint

    # show config
    config.show_config()

    


