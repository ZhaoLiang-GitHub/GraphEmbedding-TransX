# -*- coding: utf-8 -*-
# @Author: zhaoliang
# @Date:   2019-08-04 01:10:57
# @Email:  zhaoliang1@interns.chuangxin.com
# @Last Modified by:   admin
# @Last Modified time: 2019-08-04 01:12:56
import random
from config import Config
import os
class GetData(object):
    # 该类用来从文本文件中获得数据，以便后续TransX算法使用
    def __init__(self):
        self.relation_dict = {} # 用来存放关系和关系ID，每个元素是 关系：关系ID
        self.entity_dict = {} # 用来存放实体和实体ID，每个元素是 实体：实体ID
        self.triple_lists = [] # 用来存放三元组数据，每个元素是 头实体\t关系\t尾实体
        self.relation_total = 0
        self.entity_total = 0
        self.triple_total = 0
    def get_data(self, path):


        path_list = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            path_list.append(file_path)
        print(path_list)


        for i in path_list:
            if 'relation2id.txt' in i :
                relation2id_path = i
            elif 'entity2id.txt' in i :
                entity2id_path = i
            elif 'triple.txt' in i :
                triple_path = i

        try:
            with open( relation2id_path , 'r', encoding='utf-8') as f:
                a = f.readlines()
                for i in a:
                    self.relation_dict[i.strip().split('\t')[0]] = int(i.strip().split('\t')[1])
                self.relation_total = len(a)
        except:
            print('文件夹内没有relation2id.txt，\nrelation2id.txt每一行是:关系\\t关系id')
        try:
            with open( entity2id_path , 'r', encoding='utf-8') as f:
                a = f.readlines()
                for i in a:
                    self.entity_dict[i.strip().split('\t')[0]] = int(i.strip().split('\t')[1])
                self.entity_total = len(a)
        except:
            print('文件夹内没有entity2id.txt，\nentity2id.txt文件每一行是:实体\\t实体id')

        try:
            with open( triple_path , 'r', encoding='utf-8') as f:
                a = f.readlines()
                for i in a:
                    self.triple_lists.append(i.strip())
                self.triple_total = len(a)
        except:
            print('文件夹内没有triple2id.txt，\ntriple2id.txt文件每一行是:开始实体\\t结束实体\\t关系')

        return self.relation_total,self.entity_total,self.triple_total

    def get_batch(self, batch_size):
        # 从三元组的文本数据中获得批量数据
        # 在正确三元组中随机有放回抽取，这样抽取的数据占总数的 1-1/e，存在oov数据，较好的解决了过拟合
        # 不正确三元组即完全随机的抽取三个
        list_random = []
        for i in range(batch_size):
            list_random.append(random.randint(0, self.triple_total-2))
        # p :正确三元组，n :错误三元组
        # h :三元组中的头实体，r :三元组中的关系，t :三元组中的尾实体
        ph, pr, pt, nh, nr, nt = [], [], [], [], [], []  # 存放的是文字，需要转换成id
        for i in list_random:
            triple_list = self.triple_lists[i].split('\t')
            ph_i, pr_i, pt_i = triple_list[0], triple_list[2], triple_list[1]
            ph.append(int(self.word2id(ph_i)))
            pr.append(int(self.word2id(pr_i)))
            pt.append(int(self.word2id(pt_i)))
            nh_i = self.triple_lists[random.randint(0, self.triple_total-2)].split('\t')[0]
            nr_i = self.triple_lists[random.randint(0, self.triple_total-2)].split('\t')[2]
            nt_i = self.triple_lists[random.randint(0, self.triple_total-2)].split('\t')[1]
            nh.append(int(self.word2id(nh_i)))
            nr.append(int(self.word2id(nr_i)))
            nt.append(int(self.word2id(nt_i)))
        return ph, pr, pt, nh, nr, nt

    def word2id(self, word):
        try:
            return int(self.entity_dict[word])
        except :
            return int(self.relation_dict[word])


    def get_next_batch(self, batch_size):
        # 从三元组数据中依次获得批量数据
        # 正确三元组从头依次向后选择bath_size的数据
        # 错误的三元组采用随机抽取
        i = 0
        num = 0
        ph, pr, pt, nh, nr, nt = [], [], [], [], [], []  # 存放的是文字，需要转换成id
        while i < self.triple_total:
                triple_list = self.triple_lists[i].split('\t')
                ph_i, pr_i, pt_i = triple_list[0], triple_list[2], triple_list[1]
                ph.append(int(self.word2id(ph_i)))
                pr.append(int(self.word2id(pr_i)))
                pt.append(int(self.word2id(pt_i)))
                nh_i = self.triple_lists[random.randint(0, self.triple_total-2)].split('\t')[0]
                nr_i = self.triple_lists[random.randint(0, self.triple_total-2)].split('\t')[2]
                nt_i = self.triple_lists[random.randint(0, self.triple_total-2)].split('\t')[1]
                nh.append(int(self.word2id(nh_i)))
                nr.append(int(self.word2id(nr_i)))
                nt.append(int(self.word2id(nt_i)))
                num += 1
                i += 1
                if num == batch_size:
                    yield ph, pr, pt, nh, nr, nt
                    num = 0
                    ph, pr, pt, nh, nr, nt = [], [], [], [], [], []
        return ph, pr, pt, nh, nr, nt

if __name__ =='__main__' :
    pass