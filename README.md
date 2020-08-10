# Transfomer-XL

使用Tensorflow 2.3 实现CPU或单个GPU的transformer-xl，有不完善的地方后面有空再优化。

参考github：https://github.com/GaoPeng97/transformer-xl-chinese


## 编译词典
修改data_utils.py中的vocab_file文件路径，并运行生成词典文件vocab.pkl；

## 训练
run.py文件中修改全局变量VOCAB_FIL指定词典文件路径，修改DATASET指定数据集路径，执行指令
```bash
>> python run.py train
```

训练损失：
<center><img src=data/train_loss.png></center>

    
# 推理

训练模型文件
>https://drive.google.com/drive/folders/1PoPG8sTw9vDVtReO3UpVYIIor5cY07qQ?usp=sharing

执行指令：
```bash
>> python run.py inference
```

示例:
```shell
seed text >>> 人间三月天，
>> 人间三月天，年年称处称觞。殷借长江南极，後年风露细。看看朱颜青鬓，便似文章公子。平生此子胜而今，百载还同此。
```

