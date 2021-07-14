# 数智第二周培训-情感分析

#### 介绍
从本周开始我们的培训正式进入项目实战阶段，即使用Tensorflow和Keras深度学习框架解决自然语言处理中的实际问题。

本周的培训主题是情感分析，目前主流的情感分析要求对情感进行细粒度划分，即具体到喜怒哀乐恐惧惊喜等各种情感，更有甚者需要针对文本中不同的实体进行分别分析。考虑到你们当中大部分是初学者，本周的培训仅需要你们做**二分类**，分析文本是积极的还是消极的。

为了让你们体验一次完整的比赛流程，本次考核将采用**kaggle竞赛**的方式。考虑到你们当中很多人没有做这种项目的经验，我在项目中写了一个**完整的Demo**供大家参考，里面有很多注释。在这一周内，你们需要根据Demo自学文本分类的代码实现，然后想办法提高分类效果。最后，用你们自己的模型对测试集进行预测**并将预测结果按照规定的格式保存为csv文件进行提交**，网站会实时计算你们的f1得分。

为了防止你们当中的某些大佬使用预训练模型对其他人进行降维打击，我对数据集进行了**脱敏处理**，即用id替换文字，虽然数据看起来有些怪异，但是不影响考核的进行。

考核中遇到问题，可随时私聊师兄(￣▽￣)"



#### 竞赛地址

[https://www.kaggle.com/c/shuzhi-week2](https://www.kaggle.com/c/shuzhi-week2/host/settings)



#### 项目架构

│  README.md
│
├─data						# 考核所需的所有数据集
│      test.csv				# 测试集，2000条数据，不含标签
│      train.csv				# 训练集，6000条数据
│      val.csv					# 验证集，2000条数据，用于评估模型
│
└─Demo						# 代码示例，如缺库需自行安装
        cnn_model.h5		# 训练后保存的模型文件
        Figure_1.png
        log.txt					# 训练日志
        predict.py				# 预测并生成csv
        TextCNN.py			# 训练代码
        vocab.json
        **y_pred.csv			# 最终提交的结果样例**



#### 评估标准

本次考核使用的是[F1-micro](https://blog.csdn.net/lyb3b3b/article/details/84819931?utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~BlogCommendFromMachineLearnPai2~default-1.control)作为评估标准



#### 注意事项

- Deadline为7-27中午12点，**届时排名第一的选手需要做好晚上分享经验的准备**，请大家在ddl之前提交结果
- 每天最多提交4次
- 最终提交的文件必须为csv文件，其中必须拥有'id'列和'label'列，否则网页无法识别
