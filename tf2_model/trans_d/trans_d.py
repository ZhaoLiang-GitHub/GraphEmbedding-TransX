import tensorflow as tf
import numpy as np
import json
import os
import time
import pickle

TF_REQUIRED_VERSION = 2
assert(tf.__version__ >= '{}'.format(TF_REQUIRED_VERSION)) 

# 学习率
LEARNING_RATE = 0.001

# 在loss中是否加入L1正则化
L1_FLAG = True

#self.hidden_size,实体与关系的词向量长度,在知识图谱中关系数量会远远小于实体个数，所以该超参调整不能太大
EMBEDDING_DIM = 200

# 每个批度输入的三元组个数
BATCH_SIZE = 128

# 训练轮次
EPOCHS = 3

# 合页损失函数中的标准化项
MARGIN = 1.0

# 存放训练文件的路径，该路径下应该有训练时需要的三个文件，entity2id,relation2id,triple
TRIPLE_PATH = r"./data/triple_file.txt"

#输出entity_embeddings文件
ENTITY_EMBEDDINGS_PATH = r"./entity_embeddings.txt"

#输出relationship_embeddings文件
RELATIONSHIP_EMBEDDINGS_PATH = r"./relationship_embeddings.txt"

#输出data_helper文件
DATA_HELPER_PATH = r"./data_helper.bin"

#输出model文件夹
MODEL_DIR = r"./model_output"

#检查点目录
CHECK_POINT_DIR = r"./model_callback"
#检查点目录不存在则创建
if not os.path.exists(CHECK_POINT_DIR):
    os.makedirs(CHECK_POINT_DIR)

#tensorflow_board目录
TF_BOARD_DIR = os.path.join(CHECK_POINT_DIR,"logs")


class TransDModel(tf.keras.Model):
    def __init__(self,entity_total,relationship_total,embedding_dim=EMBEDDING_DIM,l1_flag=L1_FLAG,margin=MARGIN,
        entity_embeddings_file_path = ENTITY_EMBEDDINGS_PATH,relationship_embeddings_file_path = RELATIONSHIP_EMBEDDINGS_PATH):
        super().__init__()
        self.entity_total = entity_total #实体总数
        self.relationship_total = relationship_total #关系总数
        self.l1_flag = l1_flag 
        self.margin = margin
        self.entity_embeddings_file_path = entity_embeddings_file_path #存储实体embeddings的文件
        self.relationship_embeddings_file_path = relationship_embeddings_file_path #存储关系embeddings的文件
        self.ent_embeddings = tf.keras.layers.Embedding(
            input_dim=entity_total,output_dim=embedding_dim,name="ent_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        self.rel_embeddings = tf.keras.layers.Embedding(
            input_dim=relationship_total,output_dim=embedding_dim,name="rel_embedding",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        self.ent_transfer = tf.keras.layers.Embedding(
            input_dim=entity_total,output_dim=embedding_dim,name="ent_transfer",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)
        self.rel_transfer = tf.keras.layers.Embedding(
            input_dim=relationship_total,output_dim=embedding_dim,name="rel_transfer",
            embeddings_initializer=tf.keras.initializers.glorot_normal(),)

    def compute(self, h, t, r):
        return tf.math.l2_normalize(h + tf.math.reduce_sum(h * t, 1, keepdims=True) * r, 1)
    
    def call(self,pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id):
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

        pos_h_e = self.compute(pos_h_e, pos_h_t, pos_r_t)
        pos_t_e = self.compute(pos_t_e, pos_t_t, pos_r_t)
        neg_h_e = self.compute(neg_h_e, neg_h_t, neg_r_t)
        neg_t_e = self.compute(neg_t_e, neg_t_t, neg_r_t)

        if self.l1_flag:
            pos = tf.math.reduce_sum(tf.math.abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.math.reduce_sum(tf.math.abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.math.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.math.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
        
        return pos,neg

    def save_entity_relationship_embeddings(self,data_helper):
        """存储embeddings
        """
        def save_embeddings(name2id_dict,embeddings_npa,f):
            assert len(embeddings_npa) == len(name2id_dict)
            id2name_dict = {value:key for key,value in name2id_dict.items()}
            for _id,vector in enumerate(embeddings_npa):

                f.write(json.dumps([id2name_dict[_id],vector.tolist()],ensure_ascii=False) + "\n")       

        with open(self.entity_embeddings_file_path,"w") as f:
            save_embeddings(data_helper.entity_dict,self.ent_embeddings.embeddings.numpy(),f)

        with open(self.relationship_embeddings_file_path,"w") as f:
            save_embeddings(data_helper.relationship_dict,self.rel_embeddings.embeddings.numpy(),f)

@tf.function
def compute_loss(model,x):
    """计算损失
    """
    pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id = x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5]
    pos,neg = model.call(pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id)
    return tf.math.reduce_sum(tf.math.maximum(pos - neg + model.margin, 0))

class DataHelper():

    def __init__(self,triple_file_path:str):
        self.triple_file_path = triple_file_path
        self.init_raw_data()

    def init_raw_data(self):
        """保存原始数据
        """
        self.entity_dict = {} # 用来存放实体和实体ID，每个元素是 实体：实体ID
        self.entity_set = set() #用来存放所有实体
        self.entity2entity_dict = {} # 用来存放每个实体相连的边，每个元素是 实体：实体的集合
        self.relationship_dict = {} # 用来存放关系和关系ID，每个元素是 关系：关系ID
        self.triple_list_list = [] # 用来存放三元组数据，每个元素是 [头实体,关系,尾实体]
        self.relationship_total = 0
        self.entity_total = 0

        with open(self.triple_file_path) as f:
            tf.print("正在加载数据")
            i = 0
            for i,triple_list in enumerate((json.loads(v.strip()) for v in f.readlines())):
                if i % 100000 == 0:
                    tf.print("加载到第{}行".format(i))
                self.triple_list_list.append(triple_list)
                #增加实体到字典
                for entity in (triple_list[0],triple_list[2]):
                    if not entity in self.entity_dict:
                        self.entity_dict[entity] = self.entity_total
                        self.entity_total += 1
                    if not entity in self.entity2entity_dict:
                        self.entity2entity_dict[entity] = set()
                    #set会自动去重，所以每次直接添加即可
                    self.entity2entity_dict[entity].add((triple_list[0],triple_list[2]))
                    self.entity_set.add(entity)

                #增加关系到字典
                relationship = triple_list[1]
                if not relationship in self.relationship_dict:
                    self.relationship_dict[relationship] = self.relationship_total
                    self.relationship_total += 1
            tf.print("加载完成，总共{}行".format(i))
            tf.print("总共有实体{}个".format(self.entity_total))
            tf.print("总共有关系{}个".format(self.relationship_total))
    
    def word2id(self,word):
        """word2id的转化
        """
        if word in self.entity_dict:
            result = self.entity_dict[word]
        else:
            result = self.relationship_dict[word]
        return result
    
    def get_negative_entity(self,entity):
        """替换entity,获得不存在的三元组
        """
        return np.random.choice(list(self.entity_set - self.entity2entity_dict[entity]))


    def get_tf_dataset(self,batch_size = BATCH_SIZE):
        """获得训练集，验证集，测试集
        格式为:[pos_h_id,pos_t_id,pos_r_id,neg_h_id,neg_t_id,neg_r_id]
        """
        data_list = []
        tf.print("正在生成数据")
        for triple_list in self.triple_list_list:
            #每个存在的三元组要对应两个不存在的三元组，参见原文
            temp_list1 = [
                triple_list[0],triple_list[2],triple_list[1],
                self.get_negative_entity(triple_list[2]),triple_list[2],triple_list[1]
            ]
            temp_list2 = [
                triple_list[0],triple_list[2],triple_list[1],
                triple_list[0],self.get_negative_entity(triple_list[0]),triple_list[1]
            ]
            data_list.extend([[self.word2id(v) for v in temp_list1],[self.word2id(v) for v in temp_list2]])
        tf.print("生成完成")
        return tf.data.Dataset.from_tensor_slices(data_list)


def train():
    """训练
    """
    learning_rate = LEARNING_RATE
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    triple_file_path = TRIPLE_PATH
    check_point_dir = CHECK_POINT_DIR
    tf_board_dir = TF_BOARD_DIR

    data_helper_file_path = DATA_HELPER_PATH
    model_dir = MODEL_DIR


    data_helper = DataHelper(triple_file_path)
    
    model = TransDModel(data_helper.entity_total,data_helper.relationship_total)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model,)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=check_point_dir, checkpoint_name="model.ckpt", max_to_keep=None)
    
    summary_writer = tf.summary.create_file_writer(tf_board_dir)     # 实例化记录器
    tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        tf_dataset = data_helper.get_tf_dataset().shuffle(buffer_size=2**13).batch(batch_size)
        epoch_loss_avg = tf.keras.metrics.Mean()
        #训练
        for train_batch,train_x in enumerate(tf_dataset):
            with tf.GradientTape() as tape:
                loss = compute_loss(model, train_x)
                epoch_loss_avg(loss)
                tf.print("BatchSize: {} | Epoch: {:03d} | Batch: {:03d} | Loss: {:.3f},".format(batch_size,epoch,train_batch+1,loss))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

       #每个epoch输出结果
        if epoch % 1 == 0:
            tf.print("Epoch {:03d}: AverageLoss: {:.3f},".format(epoch,epoch_loss_avg.result()))
            path = checkpoint_manager.save(checkpoint_number=epoch)
            with summary_writer.as_default():                           # 指定记录器
                tf.summary.scalar("AverageLoss", epoch_loss_avg.result(), step=epoch)       # 将当前损失函数的值写入记录器
            tf.print("Save checkpoint to path: {}".format(path))
            tf.print("This epoch spends {:.1f}s".format(time.time()-start_time))

    model.save_entity_relationship_embeddings(data_helper)
    tf.saved_model.save(model,model_dir)
    with open(data_helper_file_path,"wb") as f:
        pickle.dump(data_helper,f)


def main():
    train()


if __name__ == "__main__":
    main()
