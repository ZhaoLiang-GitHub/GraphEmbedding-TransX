本仓库记录了使用TensorFlow2.0库来实现图表征学习中的Trans相关库的内容。包括TransE、TransH、TransD。有做相关工作的同学，可以通过邮件和我沟通联系 zhaoliang19960421@outlook.com
# 文件结构
- data
  - 对应的知识图谱文件夹
    - triple.txt   
    三元组文件，每一行是一个三元组，头检点分割符关系分隔符尾结点
- Config.py  
  配置文件，定义了在TransX算法中使用的超参数
- Main.py
  主文件，定义了运行函数，
- Models.py
  模型文件，定义了TransX模型和数据处理模型
  
# 运行文件
修改Config文件中的超参数，运行Main文件即可
 
