# -*- coding: utf-8 -*-
# @Author: zhaoliang
# @Date:   2019-08-04 01:10:33
# @Email:  zhaoliang1@interns.chuangxin.com
# @Last Modified by:   admin
# @Last Modified time: 2019-08-04 01:12:02

class Config(object):
    # 在知识图谱图TransX系列中的参数均在config类内修改和定义，在模型类与Main.py内不需要修改
    def __init__(self):
        self.learning_rate = 0.001 # 学习率
        self.L1_flag = True # 在loss中是否加入L1正则化
        self.hidden_size = 200 # 实体与关系的词向量长度,在知识图谱中关系数量会远远小于实体个数，所以该超参调整不能太大
        self.hidden_sizeE = 200 # 实体词向量长度，仅在TransR中使用
        self.hidden_sizeR = 10 # 关系词向量长度，仅在TransR中使用，关系向量长度与实体向量长度不宜差距太大
        self.batch_size = 100 # 每个批度输入的三元组个数
        self.epochs = 100 # 训练轮次
        self.margin = 1.0 # 合页损失函数中的标准化项
        self.relation_total = 0 # 知识图谱关系数，不需要修改，后续从输入数据中获得
        self.entity_total = 0 # 知识图谱实体数，不需要修改，后续从输入数据中获得
        self.triple_total = 0 # 知识图谱三元组个数，不需要修改，后续从数据输入中获得
        self.flie_path = './data/XunYiWenYao/' # 存放训练文件的路径，该路径下应该有训练时需要的三个文件，entity2id,relation2id,triple
        self.model_name = 'transr' # TransX模型可选[transe,transd,transh,transr],默认是TransE
        self.rel_init = None # 关系向量预训练文件，每一行是一个向量，张量大小和要训练的参数一致，仅在TransR中使用
        self.ent_init = None # 实体向量预训练文件，每一行是一个向量，张量大小和要训练的参数一致，仅在TransR中使用


