import tensorflow as tf
import numpy as np
import json
import os
import time
import pickle
import random
class DataHelper():
    '''数据处理类，该类将三元组文件进行处理，得到所有的三元组数据、实体集、关系集
    '''
    def __init__(self,config):
        self.model_name = config.model_name
        self.triple_file_path = config.triple_path
        self.entity_dict = {}  # 用来存放实体和实体ID，每个元素是 实体：实体ID
        self.entity_set = set() # 用来存放所有实体
        self.entity2entity_dict = {} # 用来存放每个实体相连的边，每个元素是 实体：实体的集合
        self.relationship_dict = {} # 用来存放关系和关系ID，每个元素是 关系：关系ID
        self.triple_list_list = [] # 用来存放三元组数据，每个元素是 [头实体,关系,尾实体]
        self.relationship_total = 0
        self.entity_total = 0
        self.head_set = {}  # 所有的头实体集合,每个元素是 实体：[以该实体为头实体的尾实体]
        self.tail_set = {}  # 所有的尾实体集合
        with open(self.triple_file_path) as f:
            print("Loading data from {}".format(self.triple_file_path))
            for line in f.readlines():
                h,r,t = line.strip().split(config.dividers)
                self.triple_list_list.append([h,r,t])
                if h not in self.head_set:
                    self.head_set[h] = [t]
                else:
                    self.head_set[h].append(t)
                if t not in self.tail_set:
                    self.tail_set[t] = [h]
                else:
                    self.tail_set[t].append(t)

                #增加实体到字典
                for _ in range(2):
                    entity = [h,t][_]  # 当前头或者尾
                    other = [h,t][1-_]  # 另一个尾或者头
                    if not entity in self.entity_dict:
                        self.entity_dict[entity] = self.entity_total
                        self.entity_total += 1
                    if not entity in self.entity2entity_dict:
                        self.entity2entity_dict[entity] = set()
                    #set会自动去重，所以每次直接添加即可
                    self.entity2entity_dict[entity].add(other)
                    self.entity_set.add(entity)
                #增加关系到字典
                if not r in self.relationship_dict:
                    self.relationship_dict[r] = self.relationship_total
                    self.relationship_total += 1
            print("总有个三元组{}个".format(len(self.triple_list_list)))
            print("总共有实体{}个".format(self.entity_total))
            print("总共有关系{}个".format(self.relationship_total))
        total_tail_per_head = 0
        total_head_per_tail = 0
        for h in self.head_set:
            total_tail_per_head += len(self.head_set[h])
        for t in self.tail_set:
            total_head_per_tail += len(self.tail_set[t])
        self.tph = 0  # 每个头实体平均几个尾实体
        self.hpt = 0  # 每个尾实体平均几个头实体
        self.tph = total_tail_per_head / len(self.head_set)
        self.hpt = total_head_per_tail / len(self.tail_set)


    def word2id(self,word):
        """word2id的转化
        """
        if word in self.entity_dict:
            result = self.entity_dict[word]
        elif word in self.relationship_dict:
            result = self.relationship_dict[word]
        else:
            exit(1)
        return result
    
    def get_negative_entity(self,entity):
        """替换entity,获得不存在的三元组
        """
        return np.random.choice(list(self.entity_set - self.entity2entity_dict[entity]))


    def get_tf_dataset(self):
        """获得训练集，验证集，测试集
        格式为:[pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id]
        """
        data_list = []
        print("Creating data")
        for triple_list in self.triple_list_list:
            # 每个存在的三元组要对应两个不存在的三元组，参见原文
            temp_list1 = [
                triple_list[0],triple_list[2],triple_list[1],
                self.get_negative_entity(triple_list[0]),triple_list[2],triple_list[1]
            ]
            temp_list2 = [
                triple_list[0],triple_list[2],triple_list[1],
                triple_list[0],self.get_negative_entity(triple_list[2]),triple_list[1]
            ]
            if self.model_name == 'transd':  
                data_list.extend([[self.word2id(v) for v in temp_list1],[self.word2id(v) for v in temp_list2]])
            elif self.model_name == 'transh':
                if random.random <(self.tph/(self.tph+self.hpt)):
                    data_list.extend([self.word2id(v) for v in temp_list1])
                else:
                    data_list.extend([self.word2id(v) for v in temp_list2])
            else: # transe or default
                if random.random <0.5: # 随机对头结点或尾结点进行伪造
                    data_list.extend([self.word2id(v) for v in temp_list1])
                else:
                    data_list.extend([self.word2id(v) for v in temp_list2])

        print("Created data")
        return tf.data.Dataset.from_tensor_slices(data_list)


class TransE(tf.keras.Model):
    '''TransE模型类，定义了TransE的参数空间和loss计算
    '''
    def __init__(self,config,data_helper):
        super().__init__()
        self.entity_total = data_helper.entity_total #实体总数
        self.relationship_total =data_helper.relationship_total #关系总数
        self.l1_flag = config.l1_flag  # L1正则化
        self.margin = config.margin  # 合页损失函数中的样本差异度值
        self.entity_embeddings_file_path = config.entity_embeddings_path #存储实体embeddings的文件
        self.relationship_embeddings_file_path = config.relationship_embeddings_path #存储关系embeddings的文件
        self.embedding_dim = config.embedding_dim  # 向量维度
        # 初始化实体语义向量空间
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.entity_total,output_dim=self.embedding_dim,name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        # 初始化关系翻译向量空间
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total,output_dim=self.embedding_dim,name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
    def compute_loss(self,x):
        # 计算一个批次数据的合页损失函数值
        # 获得头、尾、关系的 ID
        pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5]
        # 根据ID获得语义向量(E)和转移向量(T)
        pos_h_e = self.ent_embeddings(pos_h_id)
        pos_t_e = self.ent_embeddings(pos_t_id)
        pos_r_e = self.rel_embeddings(pos_r_id)

        neg_h_e = self.ent_embeddings(neg_h_id)
        neg_t_e = self.ent_embeddings(neg_t_id)
        neg_r_e = self.rel_embeddings(neg_r_id)

        if self.l1_flag:
            pos = tf.math.reduce_sum(tf.math.abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.math.reduce_sum(tf.math.abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.math.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        return tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))


class TransH(tf.keras.Model):
    '''TransE模型类，定义了TransE的参数空间和loss计算
    '''
    def __init__(self,config,data_helper):
        super().__init__()
        self.entity_total = data_helper.entity_total #实体总数
        self.relationship_total =data_helper.relationship_total #关系总数
        self.l1_flag = config.l1_flag  # L1正则化
        self.margin = config.margin  # 合页损失函数中的样本差异度值
        self.entity_embeddings_file_path = config.entity_embeddings_path #存储实体embeddings的文件
        self.relationship_embeddings_file_path = config.relationship_embeddings_path #存储关系embeddings的文件
        self.embedding_dim = config.embedding_dim  # 向量维度
        self.epsilon = config.epsilon  # 软约束中的对于法向量和翻译向量的超参
        self.C = config.C  # 软约束的参数
        # 初始化实体语义向量空间
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.entity_total,output_dim=self.embedding_dim,name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        # 初始化关系翻译向量空间
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total,output_dim=self.embedding_dim,name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        # 初始化关系超平面法向量空间
        self.norm_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total,output_dim=self.embedding_dim,name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)

    def compute_loss(self,x):
        # 计算一个批次数据的合页损失函数值
        # 获得头、尾、关系的 ID
        def _transfer(e, norm):
            # 在关系超平面上做映射，得到在关系超平面上的映射向量
            norm = tf.norm(norm,ord=2,axis=1)  # 模长为1的法向量
            return e - tf.math.reduce_sum(e * norm, 1, keepdims=True) * norm

        pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5]
        pos_h_e = self.ent_embeddings(pos_h_id)
        pos_t_e = self.ent_embeddings(pos_t_id)
        pos_r_e = self.rel_embeddings(pos_r_id)
        pos_r_n = self.norm_embeddings(pos_r_id)  # 正例关系法向量

        neg_h_e = self.ent_embeddings(neg_h_id)
        neg_t_e = self.ent_embeddings(neg_t_id)
        neg_r_e = self.rel_embeddings(neg_r_id)
        neg_r_n = self.norm_embeddings(neg_r_id)  # 负例关系法向量

        pos_h_e = _transfer(pos_h_e,pos_r_n)
        pos_t_e = _transfer(pos_t_e,pos_r_n)
        neg_h_e = _transfer(neg_h_e,neg_r_n)
        neg_t_e = _transfer(neg_t_e,neg_r_n)


        if self.l1_flag:
            pos = tf.math.reduce_sum(tf.math.abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.math.reduce_sum(tf.math.abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.math.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        

        # transH 原始论文中有软约束，但是在很多代码中都没有看到，都是直接做的正负例的合页损失函数
        hinge_loss = tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))  # 合页损失
        entity_loss = 0
        for e in self.ent_embeddings:
            embedding = self.ent_embeddings[e]
            entity_loss += tf.math.maximum(0, tf.math.reduce_sum(tf.norm(embedding,ord=2,axis=1))-1 )
        relationship_loss = 0
        for r in self.rel_embeddings:
            w = self.norm_embeddings[r] # 法向量
            d = self.rel_embeddings[r]  # 翻译向量
            relationship_loss += tf.math.maximum(0, 
                               (  tf.math.reduce_sum(tf.matmul(tf.transpose(w),d))/\
                               tf.math.reduce_sum(tf.norm(d,ord=2,axis=1))  )-self.epsilon**2)
        loss = 0
        loss += hinge_loss
        loss += self.C(entity_loss+relationship_loss)
        return loss
        '''
        # 只有正负例的合页损失函数,在很多的开源代码库中都是这么写的
        return tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))
        '''


class TrasnD(tf.keras.Model):
    '''TransD模型类,定义TransD的参数空间和loss计算
    '''
    def __init__(self,config,data_helper): 
        super().__init__()
        self.entity_total = data_helper.entity_total #实体总数
        self.relationship_total =data_helper.relationship_total #关系总数
        self.l1_flag = config.l1_flag  # L1正则化
        self.margin = config.margin  # 合页损失函数中的样本差异度值
        self.entity_embeddings_file_path = config.entity_embeddings_path #存储实体embeddings的文件
        self.relationship_embeddings_file_path = config.relationship_embeddings_path #存储关系embeddings的文件
        self.embedding_dim = config.embedding_dim  # 向量维度

        # 初始化实体语义向量空间
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=self.entity_total,output_dim=self.embedding_dim,name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        # 初始化关系翻译向量空间
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=self.relationship_total,output_dim=self.embedding_dim,name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        # 实体的转换向量
        self.ent_transfer = tf.keras.layers.Embedding(
            input_dim=self.entity_total,output_dim=self.embedding_dim,name="ent_transfer",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        # 关系的转换向量
        self.rel_transfer = tf.keras.layers.Embedding(
            input_dim=self.relationship_total,output_dim=self.embedding_dim,name="rel_transfer",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)

    
    def compute_loss(self,x):
        # 计算一个批次数据的合页损失函数值
        def _transfer(h, t, r):
            return tf.math.l2_normalize(h + tf.math.reduce_sum(h * t, 1, keepdims=True) * r, 1)
        # 获得头、尾、关系的 ID
        pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5]
        # 根据ID获得语义向量(E)和转移向量(T)
        pos_h_e = self.ent_embeddings(pos_h_id)
        pos_t_e = self.ent_embeddings(pos_t_id)
        pos_r_e = self.rel_embeddings(pos_r_id)
        pos_h_t = self.ent_transfer(pos_h_id)
        pos_t_t = self.ent_transfer(pos_t_id)
        pos_r_t = self.rel_transfer(pos_r_id)

        neg_h_e = self.ent_embeddings(neg_h_id)
        neg_t_e = self.ent_embeddings(neg_t_id)
        neg_r_e = self.rel_embeddings(neg_r_id)
        neg_h_t = self.ent_transfer(neg_h_id)
        neg_t_t = self.ent_transfer(neg_t_id)
        neg_r_t = self.rel_transfer(neg_r_id)

        pos_h_e = _transfer(pos_h_e, pos_h_t, pos_r_t)
        pos_t_e = _transfer(pos_t_e, pos_t_t, pos_r_t)
        neg_h_e = _transfer(neg_h_e, neg_h_t, neg_r_t)
        neg_t_e = _transfer(neg_t_e, neg_t_t, neg_r_t)

        if self.l1_flag:
            pos = tf.math.reduce_sum(tf.math.abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.math.reduce_sum(tf.math.abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.math.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        return tf.math.reduce_sum(tf.math.maximum(pos - neg + self.margin, 0))
