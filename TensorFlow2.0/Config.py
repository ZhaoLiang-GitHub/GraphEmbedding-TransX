import os
class Config(object):
    '''参数类，定义了在TransX算法中所需要的的超参数
    '''
    def __init__(self):
        '''General''' 
        self.model_name = 'transe'.lower()  # 模型名称，可选transe
        self.learning_rate = 0.001  # 学习率
        self.batch_size = 128  # 每个批度输入的三元组个数
        self.epochs = 3  # 训练轮次
        self.margin = 1.0  # 合页损失函数中的标准化项
        self.dividers = '\t'  # triple文件中三元组的分割符，根据实际文件进行调整
        self.triple_path = r"./data/HaoXinQing/triple.txt" # 三元组文件路径，每一行是 头实体dibiders关系dibiders尾实体
        self.save_path = r"./result"  # 保存结果的路劲
        self.entity_embeddings_path = os.path.join(self.save_path,"entity_embeddings.txt")  #输出实体向量文件
        self.relationship_embeddings_path = os.path.join(self.save_path,r"relationship_embeddings.txt") #输出关系向量
        self.data_helper_path = os.path.join(self.save_path,r"data_helper.bin" )#输出data_helper文件
        self.model_dir = os.path.join(self.save_path,r"model_output")  #输出model文件夹
        self.check_point_dir = os.path.join(self.save_path,r"model_callback" ) #检查点目录
        self.tf_board_dir = os.path.join(self.check_point_dir,"logs") #tensorflow_board目录
        # 检查点目录不存在则创建
        if not os.path.exists(self.check_point_dir):
            os.makedirs(self.check_point_dir)
        # 保运结果目录不存在就新建
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.embedding_dim = 200  # 实体与关系的词向量长度,在TransE、TransH中实体和关系的向量长度是一致的


        '''TransE'''


        '''TransH'''
        # self.norm_vector_dim = self.embedding_dim  # 关系的超平面法向量，和实体、向量长度是一致的,
        self.epsilon = 0.01  # TransH 损失函数中软约束中的对于法向量和翻译向量的超参
        self.C = 0.1  # TransH 中损失函数软约束的超参

        
        '''TransR'''
        self.entity_embedding_dim = 300  # 实体的向量长度
        self.rel_embedding_dim = 300  # 关系的向量长度，实体向量长度和关系向量长度没有关系，可自行设定


        '''CTransR'''
        # 在CTransR中对实体使用TransE进行初始化，建议为None由代码自行创建
        # 如果要使用之前有过的训练好的向量数据，向量的输入文件的格式是每一行 实体名称 分隔符 对应实体向量
        self.entity_init_path = None  
        self.rel_init_path = None  


        '''TransD'''


        


