# -*- coding: utf-8 -*-
# @Author: zhaoliang
# @Date:   2019-08-04 01:10:33
# @Email:  zhaoliang1@interns.chuangxin.com
# @Last Modified by:   admin
# @Last Modified time: 2019-08-04 19:12:02
import tensorflow as tf
from config import Config
import numpy as np
from collections import defaultdict
import os

class CalculateSimilarity(object):
    def __init__(self):
        self.dict_list = [] # 用来存储所有的实体名称
        self.dict_dict = defaultdict(list) # 用来查找对应的标签在内的所有实体
    def add_dict(self, file_dict, dict_name):
        """
        构建字典
        :param file_dict: 字典数据文件名，其中每一行为一个实体 元素是【实体，类别】
        :param name: 这类实体的名字，在以后用于关系抽取
        :return:
        """
        A = []
        B = []
        with open(file_dict, 'r', encoding='utf-8') as f:
            for i in f.readlines():
                if i.strip() != '':
                    A.append([i.strip(), dict_name])
                    B.append(i.strip())
        A.sort(key=lambda x: len(x), reverse=True)
        self.dict_dict[dict_name].extend(B)
        self.dict_list.extend(A)


    def _get_data(self,path):
        relation_dict = {}  # 用来存放关系和关系ID，每个元素是 关系：关系ID
        entity_dict = {}  # 用来存放实体和实体ID，每个元素是 实体：实体ID

        path_list = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            path_list.append(file_path)
        print('输入的文件夹内共有文件：',' '.join(path_list))


        for i in path_list:
            if 'relation2id.txt' in i :
                relation2id_path = i
            elif 'entity2id.txt' in i :
                entity2id_path = i
            elif 'triple.txt' in i :
                triple_path = i

        try:
            with open(relation2id_path, 'r', encoding='utf-8') as f:
                a = f.readlines()
                for i in a:
                    relation_dict[i.strip().split('\t')[0]] = int(i.strip().split('\t')[1])
        except:
            print('文件夹内没有relation2id.txt，\nrelation2id.txt每一行是:关系\\t关系id')

        try:
            with open(entity2id_path, 'r', encoding='utf-8') as f:
                a = f.readlines()
                for i in a:
                    entity_dict[i.strip().split('\t')[0]] = int(i.strip().split('\t')[1])
        except:
            print('文件夹内没有entity2id.txt，\nentity2id.txt文件每一行是:实体\\t实体id')

        return relation_dict, entity_dict


    def _calculate_distance(self, vector1, vector2):
        cosine_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2))) # 余弦夹角
        euclidean_distance = np.sqrt(np.sum(np.square(vector1-vector2))) # 欧式距离
        return cosine_distance


    def main(self,topk):
        sess = tf.Session()
        config = Config()
        # import_meta_graph填的名字meta文件的名字
        saver = tf.train.import_meta_graph('./模型保存路径/XunYiWenYao/TransR/model.ckpt.meta')
        # 检查checkpoint，所以只填到checkpoint所在的路径下即可，不需要填checkpoint
        saver.restore(sess, tf.train.latest_checkpoint("./模型保存路径/XunYiWenYao/TransR"))
        ent_embedding = sess.run('ent_embedding:0') # 经过表征学习得到的实体向量矩阵，每一行是一个实体
        rel_embedding = sess.run('rel_embedding:0') # 经过表征学习得到的关系向量矩阵，每一行是一个关系
        relation_dict, entity_dict = self._get_data(config.flie_path)
        max_similarity = 0
        max_similarity_tuple = []
        for k1,v1 in entity_dict.items():

            k1_label = None
            for dict in self.dict_list:
                if k1 == dict[0]:
                    k1_label = dict[1] # 找到该实体所在标签的名字，以便在之后的循环中只找这个实体标签内的实体
                    break
            if k1_label == None:
                # print('不在实体列表内，需要人工筛查',k1)
                continue


            if k1_label != '疾病':
                continue


            similarity_topk = defaultdict(lambda x:0)
            for k2,v2 in entity_dict.items():
                if k1 != k2 and k2 in self.dict_dict[k1_label]:
                # if k1 != k2  :
                    vector1 = ent_embedding[v1]
                    vector2 = ent_embedding[v2]
                    distance = self._calculate_distance(vector1,vector2)
                    similarity_topk[k2] = distance
                else:
                    similarity_topk[k2] = 0
            similarity_topk = sorted(similarity_topk.items(),key= lambda  x:x[1],reverse=True)
            if similarity_topk[0][1] == 0:
                print(k1,'没有找到相同实体标签内的想相近实体')
                continue
            if similarity_topk[0][1] > max_similarity:
                max_similarity = similarity_topk[0][1]
                max_similarity_tuple = [ k1,similarity_topk[0][0]   ]
            print(k1.strip(),'所属实体类别是{},和他最相似的实体是:'.format(k1_label))
            for i in range(topk):
                print('\t',similarity_topk[i][0])
            print('\n\n')
        print('在数据中最大的相似度是{}'.format(max_similarity))
        print('在数据中最相似的两个本体是{}'.format('、'.join(max_similarity_tuple)))



if __name__ == '__main__':
    caculatesimilarity = CalculateSimilarity()

    caculatesimilarity.add_dict('../KnowledegGraph/data/dict/中成药成份.txt', '中药成份')
    caculatesimilarity.add_dict('../KnowledegGraph/data/dict/西药成份.txt', '西药成份')
    caculatesimilarity.add_dict('../KnowledegGraph/data/dict/病人属性.txt', '病人属性')
    caculatesimilarity.add_dict('../KnowledegGraph/data/dict/疾病.txt', '疾病')
    caculatesimilarity.add_dict('../KnowledegGraph/data/dict/症状.txt', '症状')
    caculatesimilarity.add_dict('../KnowledegGraph/data/dict/中医证型.txt', '中医证型')
    caculatesimilarity.add_dict('../KnowledegGraph/data/haoxinqing/好心情中的人群.txt', '病人属性') # 添加了在好心情中特有的人群
    caculatesimilarity.add_dict('../KnowledegGraph/data/haoxinqing/好心情中的检查.txt', '检查手段') # 添加了在好心情中特有的检查
    caculatesimilarity.add_dict('../KnowledegGraph/data/haoxinqing/好心情中的疾病.txt', '疾病')    # 添加了在好心情中特有的疾病
    caculatesimilarity.add_dict('../KnowledegGraph/data/haoxinqing/好心情中的症状.txt', '症状')    # 添加了在好心情中特有的症状
    caculatesimilarity.add_dict('../KnowledegGraph/data/haoxinqing/好心情中的证型.txt', '证型')    # 添加了在好心情中特有的症状
    caculatesimilarity.add_dict('../KnowledegGraph/data/haoxinqing/好心情中的西药成份.txt', '西药成份') # 添加了在好心情中特有的成份
    caculatesimilarity.add_dict('../KnowledegGraph/data/haoxinqing/好心情中的中药成份.txt', '中药成份') # 添加了在好心情中特有的成份

    caculatesimilarity.main(topk=5)