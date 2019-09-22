# -*- coding: utf-8 -*-
# @Author: Zhiwei Wu
# @Date: 2019/9/20 20:41
# @contact: zhiwei.w@qq.com
import torch.utils.data as torch_data


class dataset(torch_data.Dataset):
    def __init__(self, src, tgt, raw_src, raw_tgt):

        self.src = src
        self.tgt = tgt
        self.raw_src = raw_src
        self.raw_tgt = raw_tgt

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], \
               self.raw_src[index], self.raw_tgt[index]

    def __len__(self):
        return len(self.src)