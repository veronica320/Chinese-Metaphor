# Chinese-Metaphor
CCL 2018 Shared Task - 中文隐喻识别与情感分析

## Task Description
* 任务细节: http://ir.dlut.edu.cn/news/detail/508
* Update: 子任务一为二分类任务，只需辨别是否是动词隐喻即可
* 时间：9.30截止。每支队伍可于9月9日、9月16日、9月23日、9月30日，截止每周日晚上十点提交结果；每支队伍在每个星期最多仅可提交三次，并按照最后提交的结果计算排名。于9月10日、17日、24日、10月1日下午五点前公布在网址（http://ir.dlut.edu.cn/）。
* 训练数据: http://ir.dlut.edu.cn/File/Download?cid=3 《CCL 2018 中文隐喻识别与情感分析评测数据》
* 测试数据（无标签）: http://ir.dlut.edu.cn/File/Download?cid=3 《CCL 2018 中文隐喻识别与情感分析测试数据》
* 提醒：按组织方要求，该数据集仅可为本次评测任务使用，其它情况下使用需与组织方联系。

## Repo Structure
* /Corpus: 存储Penn StateUCMC中文隐喻语料库（暂时用不到）
* /data: 训练和测试数据
* /dicts: 两个子任务的关系词典，以及词汇表
* /memo: 会议记录
* /model_structure: nn模型的结构图
* /paper：相关文献
* /pretrained_emb: 网上下载的预训练word embedding（基于wikipedia），已过滤好
* /src：代码
* /results: 模型评测结果和生成的测试标签
* /models: 需要自己建一下这个路径，下面设/verb和/emo两个子路径，用于存放训练好的模型
* /submission: 提交的结果文件，按日期存放

## Code Structure
1. 核心代码：
      1. conf.py: 设置各种参数
      2. multi_cgru_keras.py: 模型结构
      3. train.py: 在90%的训练数据上训练模型
      4. eva_model.py：在10%的训练数据上评测模型表现
      5. generate_test_labels.py: 在测试集上预测标签
2. 辅助代码:
      1. split_data.py: 把训练集分成90%（用于训练）和10%（用于评测模型表现）
      2. back_translate.py: 用Google Translate API来增加训练数据
      3. convert_data.py: 把数据从xml转换成txt，把数字标签转换为易理解的文字标签
      4. data_provider.py: 读取数据，为训练做准备
      5. filter_wordemb.py: 基于train和test data过滤预训练词向量，只保留data中出现的词（目前的wiki词向量已过滤好）

## How to run code
1. 在conf.py中设置相关参数
2. 运行train.py，训练模型
3. 运行eva_model.py, 评估模型表现
4. 根据第三步中的评估结果，选出表现较好的模型用generate_test_labels生成测试数据标签

## Done
1. NN Baseline: 基于CGRU，最好表现(accuracy)task1约70%，task2约39%
    1. 对比：Majority Baseline，task2 37%
    2. 对比：基于情感词库的Naive baseline，不用机器学习，task2 51%
2. 基于NN Bseline，尝试以下feature：
    1. 优化Embedding层
        1. 用pre-trained embedding替代模型自己学习的embedding，task2最好表现约acc 50%
        2. 字词向量拼接：配合减小smooth参数，task2 macro f - 39.6%
    2. Back Translation
        1. Google Translate 6种语言，实验了几种过滤方法，task2最好表现约acc 53%
    3. 其他模型结构
        1. 直接使用Embedding作为分类特征
        2. LSTM+fully connected：task2 macro f - 40%
3. 一点Error analysis：
    1. 观察到overfitting严重，故尝试调整l2(↑), dropout(↑), smooth(↓)，但并未发现大的改变；同时发现同一模型表现不稳定（task2多次运行差距可达10%）
    2. Bad case其中有一部分是有转折的句子（e.g. 包含“怎么可能没”“无法”“既然”等词语）
    3. 发现数据中一部分标注存疑
4. 获取Penn State中文隐喻语料库，可用于自训练word embedding
5. 补充训练语料：用其他英文语料翻译回来，补充训练语料
6. 调参

## Todolist
1. 基于NN baseline尝试更多feature:
    1. 继续优化Embedding层
        1. 使用其他pre-trained embedding: e.g. 基于Penn State隐喻语料库训练的embedding, ELMo Embedding等
    2. 将情感词库加入nn:
        1. 对label做embedding：现有方法只用于有递进关系的labels（very neg, neg, neutral, pos, very pos）
    3. 动词、名词的subcategory
    4. Dependency relation
    5. 通过观察数据，考察虚词在两个子任务中起到的作用，再决定将虚词的哪些信息加入模型。虚词：什么样的信息有帮助？
2. 尝试其他模型结构：
    1. （参考'Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms'一文）
    2. 使用Transformer最为sentence encoder（参见'Attention Is All You Need'一文)

## Resources
1. Penn State中文隐喻语料库(http://www.personal.psu.edu/xxl13/download.html)
2. 大连理工情感词汇本体库(http://ir.dlut.edu.cn/EmotionOntologyDownload)

## Organizor
大连理工大学信息检索研究室
* Email: 林鸿飞 irlab@dlut.edu.cn
* QQ group: 604185567
