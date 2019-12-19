# tf2版本，主要修改如下：

## 1.完全由tf2撰写，继承了tensorboard，运行程序后直接运行run_tensorboard.sh即可

## 2.数据文件只保留了triple文件，格式为[entity0,relationship01,entity1],id由程序生成;格式由 \t 分割变为json.dumps(list),这样即使在实体词内有\t时也可正常运行

## 3.没有抽取公共类，确保使用者可以直接拿走整个模型做修改

## 4. todo : 除transD外的其他模型
