# -*- coding: utf-8 -*-
# @Author: Zhiwei Wu
# @Date: 2019/9/21 10:04
# @contact: zhiwei.w@qq.com
import numpy as np
from tqdm import tqdm
import json

def rewrite_train():
    out_file = open('../data/NLPCC/retrain.txt', 'w', encoding='utf8')
    with open('../data/NLPCC/train.txt', 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            line = json.dumps(line,ensure_ascii=False)
            out_file.write(line + '\n')
    out_file.close()


def write_lt(pgh_lt, save_file):
    with open(save_file, 'w', encoding='utf-8') as f:
        for pgh in pgh_lt:
            f.write(pgh+'\n')

def static_len(pgh_lt, name):
    print('{} len mean, 0.9 and 0.8'.format(name))
    total_len = map(len, pgh_lt)
    total_len = sorted(total_len, reverse=False)
    print(np.mean(total_len))
    print(total_len[int(len(total_len)*0.9)])
    print(total_len[int(len(total_len)*0.8)])

def prepare_NLPCC(input_file, src_file, tgt_file):
    srcs = []
    tgts = []
    count = 1
    with open(input_file, 'r', encoding='utf-8') as f_:
        line_count = 1
        for line in f_.readlines():
            data = json.loads(line)
            article = data['article'].strip()
            summ = data['summarization'].strip()
            if article == "" or summ == "":
                print('at line {} summ or art be none')
                continue
            srcs.append(article)
            tgts.append(summ)
            line_count += 1

    print('number of srcs {}, tgts {}'.format(len(srcs), len(tgts)))

    static_len(pgh_lt=srcs, name='srcs')
    static_len(pgh_lt=tgts, name='tgts')

    write_lt(srcs, src_file)
    write_lt(tgts, tgt_file)
    print('prepare_NLPCC end')

"""
train set:
number: 50000
srcs len mean, 0.9 and 0.8
1036.16026
2174
1484
tgts len mean, 0.9 and 0.8
45.06544
57
53

dev set
number of srcs 2000, tgts 2000
srcs len mean, 0.9 and 0.8
1037.078
2054
1447
tgts len mean, 0.9 and 0.8
44.9165
57
52
"""


if __name__ == '__main__':
    # rewrite_train()
    prepare_NLPCC('../data/NLPCC/train.txt', '../data/NLPCC/train.src', '../data/NLPCC/train.tgt')
    prepare_NLPCC('../data/NLPCC/valid.txt', '../data/NLPCC/valid.src', '../data/NLPCC/valid.tgt')
