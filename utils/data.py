
import numpy as nps
from utils.functions import read_config
import torch.utils.data

class Data():
    def __init__(self, config):

        self.checkpoints = None
        
        # checkpoint
        if config.restore:
            print('loading checkpoint...\n')
            checkpoints = torch.load(opt.restore)
            config = checkpoints['config']

        self.config = config
        # data
        print('loading data...\n')
        start_time = time.time()
        datas = torch.load(config.data)
        print('loading time cost: %.3f' % (time.time()-start_time))

        trainset, validset = datas['train'], datas['valid']
        src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']

        if config.limit > 0:
            trainset = trainset.src[:opt.limit]
            validset = trainset
    
        # build dataloader
        self.trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=0,
                                              collate_fn=self.padding)
        self.validloader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=self.padding)

        # pretrain emb
        pretrain_embed = torch.load(config.emb_file) if config.emb_file else None

        

    def padding(self, data):
         #data.sort(key=lambda x: len(x[0]), reverse=True)
        src, tgt, raw_src, raw_tgt = zip(*data)

        src_len = [len(s) for s in src]
        src_pad = torch.zeros(len(src), max(src_len)).long()
        for i, s in enumerate(src):
            end = src_len[i]
            src_pad[i, :end] = s[:end]

        tgt_len = [len(s) for s in tgt]
        tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
        for i, s in enumerate(tgt):
            end = tgt_len[i]
            tgt_pad[i, :end] = s[:end]

        batch = {}
        batch['raw_src'] = raw_src
        batch['raw_tgt'] = raw_tgt
        batch['src'] = src_pad.t()
        batch['tgt'] = tgt_pad.t()
        batch['src_len'] = torch.LongTensor(src_len)
        batch['tgt_len'] = torch.LongTensor(tgt_len)

        return batch

        