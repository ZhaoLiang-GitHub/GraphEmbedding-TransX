# -*- coding: utf-8 -*-
# @Author: zhaoliang
# @Date:   2019-08-04 01:03:50
# @Email:  zhaoliang1@interns.chuangxin.com
# @Last Modified by:   admin
# @Last Modified time: 2019-08-05 01:08:31
import tensorflow as tf
import numpy as np

class TransE(object):

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.margin = config.margin
        self.learning_rate = config.learning_rate
        self.L1_flag = config.L1_flag
        self.epochs = config.epochs
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.triple_total = config.triple_total

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.compat.v1.get_variable(name="ent_embedding", shape=[self.entity_total, self.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.compat.v1.get_variable(name="rel_embedding", shape=[self.relation_total, self.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))


        self.pos_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_r = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_r = tf.compat.v1.placeholder(tf.int32, [None])
        pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
        pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
        pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
        neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
        neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

        if self.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos
        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))


class TransD(object):

    def calc(self, h, t, r):
        return tf.nn.l2_normalize(h + tf.reduce_sum(h * t, 1, keepdims=True) * r, 1)

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.margin = config.margin
        self.learning_rate = config.learning_rate
        self.L1_flag = config.L1_flag
        self.epochs = config.epochs
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.triple_total = config.triple_total

        self.pos_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_r = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_r = tf.compat.v1.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.compat.v1.get_variable(name="ent_embedding", shape=[self.entity_total, self.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.compat.v1.get_variable(name="rel_embedding", shape=[self.relation_total, self.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.ent_transfer = tf.compat.v1.get_variable(name="ent_transfer", shape=[self.entity_total, self.hidden_size],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_transfer = tf.compat.v1.get_variable(name="rel_transfer", shape=[self.relation_total, self.hidden_size],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
        pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
        pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
        pos_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_h)
        pos_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.pos_t)
        pos_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.pos_r)

        neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
        neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
        neg_h_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_h)
        neg_t_t = tf.nn.embedding_lookup(self.ent_transfer, self.neg_t)
        neg_r_t = tf.nn.embedding_lookup(self.rel_transfer, self.neg_r)

        pos_h_e = self.calc(pos_h_e, pos_h_t, pos_r_t)
        pos_t_e = self.calc(pos_t_e, pos_t_t, pos_r_t)
        neg_h_e = self.calc(neg_h_e, neg_h_t, neg_r_t)
        neg_t_e = self.calc(neg_t_e, neg_t_t, neg_r_t)


        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))
            

class TransH(object):
    def calc(self, e, n):
        norm = tf.nn.l2_normalize(n, 1)
        return e - tf.reduce_sum(e * norm, 1, keepdims=True) * norm
    
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.margin = config.margin
        self.learning_rate = config.learning_rate
        self.L1_flag = config.L1_flag
        self.epochs = config.epochs
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.triple_total = config.triple_total

        self.pos_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.pos_r = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_h = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_t = tf.compat.v1.placeholder(tf.int32, [None])
        self.neg_r = tf.compat.v1.placeholder(tf.int32, [None])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.compat.v1.get_variable(name="ent_embedding", shape=[self.entity_total, self.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.compat.v1.get_variable(name="rel_embedding", shape=[self.relation_total, self.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.normal_vector = tf.compat.v1.get_variable(name="normal_vector", shape=[self.relation_total, self.hidden_size],
                                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)

            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

            pos_norm = tf.nn.embedding_lookup(self.normal_vector, self.pos_r)
            neg_norm = tf.nn.embedding_lookup(self.normal_vector, self.neg_r)

            pos_h_e = self.calc(pos_h_e, pos_norm)
            pos_t_e = self.calc(pos_t_e, pos_norm)
            neg_h_e = self.calc(neg_h_e, neg_norm)
            neg_t_e = self.calc(neg_t_e, neg_norm)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))


class TransR(object):
    def __init__(self, config, ent_init = None, rel_init = None):

        self.batch_size = config.batch_size
        self.margin = config.margin
        self.learning_rate = config.learning_rate
        self.L1_flag = config.L1_flag
        self.epochs = config.epochs
        self.entity_total = config.entity_total
        self.relation_total = config.relation_total
        self.triple_total = config.triple_total
        self.sizeE = config.hidden_sizeE
        self.sizeR = config.hidden_sizeR

        ''''
        在原始论文中TransR采用的提前预训练向量，然后在一个知识图谱中进行finetuning,所以如果有提前预训练的模型参数
        '''
        with tf.name_scope("read_inputs"):
            self.pos_h = tf.compat.v1.placeholder(tf.int32, [self.batch_size])
            self.pos_t = tf.compat.v1.placeholder(tf.int32, [self.batch_size])
            self.pos_r = tf.compat.v1.placeholder(tf.int32, [self.batch_size])
            self.neg_h = tf.compat.v1.placeholder(tf.int32, [self.batch_size])
            self.neg_t = tf.compat.v1.placeholder(tf.int32, [self.batch_size])
            self.neg_r = tf.compat.v1.placeholder(tf.int32, [self.batch_size])

        with tf.name_scope("embedding"):
            if ent_init != None:
                self.ent_embeddings = tf.Variable(np.loadtxt(ent_init), name="ent_embedding", dtype=np.float32)
            else:
                self.ent_embeddings = tf.compat.v1.get_variable(name="ent_embedding", shape=[self.entity_total, self.sizeE],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            if rel_init != None:
                self.rel_embeddings = tf.Variable(np.loadtxt(rel_init), name="rel_embedding", dtype=np.float32)
            else:
                self.rel_embeddings = tf.compat.v1.get_variable(name="rel_embedding", shape=[self.relation_total, self.sizeR],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        rel_matrix = np.zeros([self.relation_total, self.sizeR * self.sizeE], dtype=np.float32)
        for i in range(self.relation_total):
            for j in range(self.sizeR):
                for k in range(self.sizeE):
                    if j == k:
                        rel_matrix[i][j * self.sizeE + k] = 1.0
        self.rel_matrix = tf.Variable(rel_matrix, name="rel_matrix")

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h), [-1, self.sizeE, 1])
            pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t), [-1, self.sizeE, 1])
            pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r), [-1, self.sizeR])
            neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h), [-1, self.sizeE, 1])
            neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t), [-1, self.sizeE, 1])
            neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r), [-1, self.sizeR])
            pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.pos_r), [-1, self.sizeR, self.sizeE])
            neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.neg_r), [-1, self.sizeR, self.sizeE])

            pos_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_h_e), [-1, self.sizeR]), 1)
            pos_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_t_e), [-1, self.sizeR]), 1)
            neg_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_h_e), [-1, self.sizeR]), 1)
            neg_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_t_e), [-1, self.sizeR]), 1)

        if config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0))

