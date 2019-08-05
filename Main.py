# -*- coding: utf-8 -*-
# @Author: zhaoliang
# @Date:   2019-08-04 01:05:34
# @Email:  zhaoliang1@interns.chuangxin.com
# @Last Modified by:   admin
# @Last Modified time: 2019-08-04 01:12:57
import tensorflow as tf
import os
import re
from config import Config
from getdata import GetData
from Models import TransE,TransD,TransH,TransR


def main(argv=None):
	config = Config()
	KG_name = config.flie_path.split('/')[-2]
	getdata = GetData()
	config.relation_total, config.entity_total, config.triple_total = getdata.get_data(config.flie_path)

	if config.model_name.lower() == 'transe':
		trainModel = TransE(config=config)
	elif config.model_name.lower() == 'transd':
		trainModel = TransD(config=config)
	elif config.model_name.lower() == 'transh':
		trainModel = TransH(config = config)
	elif config.model_name.lower() == 'transr':
		trainModel = TransR(config=config)
	else:
		trainModel = TransE(config=config)
		print('输入TransX模型名称有误，默认采用TransE模型')

	with tf.compat.v1.Session() as sess:
		train_op = tf.compat.v1.train.GradientDescentOptimizer(trainModel.learning_rate).minimize(trainModel.loss)
		saver = tf.compat.v1.train.Saver()
		sess.run(tf.compat.v1.global_variables_initializer())
		next_batch = getdata.get_next_batch(trainModel.batch_size)
		min_loss = 0
		gloabl_step = 0
		for epoch in range(trainModel.epochs):
			# 有放回的随机采样
			pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch = getdata.get_batch(trainModel.batch_size)

			# 按批次依次抽取数据
			# pos_h_batch, pos_r_batch, pos_t_batch, neg_h_batch, neg_r_batch, neg_t_batch = next_batch.__next__()

			feed_dict = {
				trainModel.pos_h: pos_h_batch,
				trainModel.pos_t: pos_t_batch,
				trainModel.pos_r: pos_r_batch,
				trainModel.neg_h: neg_h_batch,
				trainModel.neg_t: neg_t_batch,
				trainModel.neg_r: neg_r_batch
			}
			sess.run([trainModel.loss, train_op], feed_dict=feed_dict)
			loss = sess.run(trainModel.loss, feed_dict=feed_dict)
			if loss<min_loss:
				min_loss = loss
				gloabl_step = epoch
			if epoch % 50 == 0:
				print('epoch:', epoch, ',loss:', loss)
		saver_add = './模型保存路径/' + KG_name + '/' +str(type(trainModel)).replace("<class 'Models.",'').replace("'>",'')+'/'
		print('模型文件保存在',saver_add+'model.ckpt')
		try:
			os.makedirs(saver_add)
		except:
			pass
		saver.save(sess, saver_add+'model.ckpt')


if __name__ == '__main__':
	tf.compat.v1.app.run()