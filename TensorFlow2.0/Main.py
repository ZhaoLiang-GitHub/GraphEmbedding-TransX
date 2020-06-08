#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :GraphEmbedding TransX 系列算法的主文件，在主文件内定义了一个主函数类，
         在这个类内实例化了需要的数据类和模型类，利用数据类对数据进行处理，
         利用模型类训练一个基于翻译模型的图表征模型，将该模型计算得到的实体和关系向量分别保存下来；
         定义了一个模型评估类，用三元组分类来评估模型好坏，用相似度来找到最接近的实体
@Time   :2020/06/08 16:30:37
@Author :zhaoliang19960421@outlook.com
'''

import tensorflow as tf
import time
import pickle
import Config
import Model
import json
class GraphEmbedding_TranX(object):
    def run(self):
        """主函数，在主函数内实例化类，获得数据，进行训练得到图表征结果
        """
        TF_REQUIRED_VERSION = 2  
        assert(tf.__version__ >= '{}'.format(TF_REQUIRED_VERSION))  # 判断当前的tf版本是否是2.X版本
        
        self.config = Config.Config()  # 超参数类

        self.data_helper = Model.DataHelper(self.config)  # 数据处理类
        # 模型类
        if self.config.model_name == 'tranr':
            self.model = Model.TrasnR(self.config,self.data_helper)  
        elif self.config.model_name == 'trand':
            self.model = Model.TrasnD(self.config,self.data_helper)  
        else : # transe or default
            self.model = Model.TrasnE(self.config,self.data_helper)  
        
        optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)  # 优化器
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=self.model,)  # 检查点
        checkpoint_manager = tf.train.CheckpointManager(
                                            checkpoint, directory=self.config.check_point_dir, 
                                            checkpoint_name="model.ckpt", max_to_keep=1)  # 优化检查点
        
        summary_writer = tf.summary.create_file_writer(self.config.tf_board_dir) # 实例化记录器
        tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            tf_dataset = self.data_helper.get_tf_dataset().shuffle(buffer_size=1000)\
                                                            .batch(self.config.batch_size)
            epoch_loss_avg = tf.keras.metrics.Mean()
            #训练
            for train_batch,train_x in enumerate(tf_dataset):
                with tf.GradientTape() as tape:
                    loss = self.model.compute_loss(train_x)
                    epoch_loss_avg(loss)
                    print("BatchSize: {} | Epoch: {:03d} | Batch: {:03d} | Loss: {:.3f},"\
                                .format(self.config.batch_size,epoch,train_batch+1,loss))
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

           #每个epoch输出结果
            if epoch % 1 == 0:
                print("Epoch {:03d}: AverageLoss: {:.3f},".format(epoch,epoch_loss_avg.result()))
                path = checkpoint_manager.save(checkpoint_number=epoch)
                with summary_writer.as_default(): # 指定记录器
                    tf.summary.scalar("AverageLoss", epoch_loss_avg.result(), step=epoch)  
                    # 将当前损失函数的值写入记录器
                print("Save checkpoint to path: {}".format(path))
                print("This epoch spends {:.1f}s".format(time.time()-start_time))
        tf.saved_model.save(self.model,self.config.model_dir)


    def save_entity_relationship_embeddings(self):
        """存储embeddings
        """
        def save_embeddings(name2id_dict,embeddings_npa,f):
            assert len(embeddings_npa) == len(name2id_dict)
            id2name_dict = {value:key for key,value in name2id_dict.items()}
            for _id,vector in enumerate(embeddings_npa):
                f.write(json.dumps([id2name_dict[_id],vector.tolist()],ensure_ascii=False) + "\n")       

        with open(self.config.entity_embeddings_path,"w") as f:
            save_embeddings(self.data_helper.entity_dict,
                            self.model.ent_embeddings.embeddings.numpy(),f)

        with open(self.config.relationship_embeddings_path,"w") as f:
            save_embeddings(self.data_helper.relationship_dict,
                            self.model.rel_embeddings.embeddings.numpy(),f)

        with open(self.config.data_helper_path,"wb") as f:
            pickle.dump(self.data_helper,f)

    def evaluation_model(self):
        # TODO
        '''评估当前的图表征学习模型的好坏
        获得所有的正例三元组，计算三元组之间的势能差值，并打上标签1
        将正例三元组按照对应的取负例方法得到负例三元组，计算负例三元组的势能差值并打上标签0
        把获得的所有的正负例三元组划分成训练接和测试集，训练个分类器（LR）
        在测试集上看分类准确率，把该准确率作为该图表征模型的表征能力
        '''
        pass

    def most_similar_entity(self,topk=5):
        #TODO
        '''利用图表征之后的向量计算每个实体最相似的前topk个实体
        如果有实体的分类标准，则可以通过数据分类文本来获得实体分类
        例如 头疼、感冒都是疾病，双黄连、青霉素都是药品
        那么在计算头疼最相近的实体时只会去疾病分类下的实体中寻找，而不会去药品下寻找
        '''
        pass

if __name__ == "__main__":
    graphembeddingtranx = GraphEmbedding_TranX()
    graphembeddingtranx.run()  # 运行图表征，获得表征结果
    graphembeddingtranx.save_entity_relationship_embeddings()  # 把实体关系的 名词：向量保存下来
