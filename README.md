# GraphEmbedding-TransX
使用TensorFlow实现对知识图谱的表征学习，包括transe\transd\transh\transr
如果有人在做相关工作的话，可以邮件与我联系1318525510@qq.c0m

## 依赖库
    
1. TensorFlow 1.14 
    
    注：tf1.0和tf2.0有一些函数的命名发生了变化，
        在该项目代码中已经按照提醒将即将过期的方法名修改为了最新的名字，
        如果在使用中出现了代码提醒或者在更新的tf版本中导致某些函数不可使用，请按照最新的tf函数名进行修改，并提交request

2. numpy 1.16

## 参数设置

在config.py文件中对模型参数进行设置，在models.py与main.py文件中的参数均不需要修改，如果在使用代码中出现未知错误，请在config.py中修改参数并提交request

## 文件内容

1. ./data/KG_name/entity2id.txt
    
    该文件是实体与实体ID的映射，每一行是 实体\t实体ID
2. ./data/KG_name/relation2id.txt

    该文件是关系与关系ID的映射，每一行是 关系\t关系ID
3. ./data/KG_name/triple.txt

    该文件是在知识图谱中三元组数据，每一行是 头实体\t尾实体\t关系
    注：默认知识图谱中用三元组来表示，如果知识图谱中的表达比三元组要更加丰富，需要在模型中增加一个向量并修改loss的计算方法
4. ./config.py

    参数设置文件
    
5. ./Models.py 

    模型文件，其中共有transe\transd\trasnh\transr四个模型类，每个类均是独立的，如果出现未知错误仅需修改对应类并提交request
6. ./get_parameter.py   

    在训练完成之后加载模型参数以便使用
6. ./Main.py

    主文件，修改config.py文件中的参数设置，运行该文件即可

