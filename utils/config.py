import yaml
import os

class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self
        # cuda
        self.device = config.device if torch.cuda.is_availabel() else 'cpu'
    
    def show_config(self):
        print('='*10 + ' Config setting ' + '='*10 )
        for name,value in self.__dict__.items():
            print('\t {}: {}'.format(name,value))


def read_config(path):
    return Config(yaml.load(open(path, 'r')))
