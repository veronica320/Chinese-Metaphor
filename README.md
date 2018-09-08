# Chinese-Metaphor
CCL 2018 Shared Task - 中文隐喻识别与情感分析

## Task Description
* 任务细节: http://ir.dlut.edu.cn/news/detail/508
* 训练数据: http://ir.dlut.edu.cn/File/Download?cid=3 《CCL 2018 中文隐喻识别与情感分析评测数据》
* 测试数据（无标签）: http://ir.dlut.edu.cn/File/Download?cid=3 《CCL 2018 中文隐喻识别与情感分析测试数据》
* 提醒：按组织方要求，该数据集仅可为本次评测任务使用，其它情况下使用需与组织方联系。


## Organizor
大连理工大学信息检索研究室
* Email: 林鸿飞 irlab@dlut.edu.cn
* QQ group: 604185567

## Code Structure

## Done
1. NN Baseline: 基于CGRU，最好表现(accuracy)task1约70%，task2约39%
    1. 对比：Majority Baseline，task2 37%
    2. 对比：基于情感词库的Naive baseline，不用机器学习，task2 51%
2. 基于NN Bseline，尝试以下feature：
    1. 优化Embedding层
        1. 用pre-trained embedding替代模型自己学习的embedding，task2最好表现约50%
    2. Back Translation
        1. Google Translate 6种语言，实验了几种过滤方法，task2最好表现约53%
3. 一点Error analysis：
    1. 观察到overfitting严重，故尝试调整l2(↑), dropout(↑), smooth(↓)，但并未发现大的改变；同时发现同一模型表现不稳定（task2多次运行差距可达10%）
    2. Bad case其中有一大部分是有转折的句子（e.g. 包含“怎么可能没”“无法”“既然”等词语）
    3. 发现数据中一部分标注存疑
4. 获取Penn State中文隐喻语料库，可用于自训练word embedding

## Todolist
1. 基于NN baseline尝试更多feature:
    1. 继续优化Embedding层
        1. 字词向量拼接
        2. 使用其他pre-trained embedding: e.g. 基于Penn State隐喻语料库训练的embedding, ELMo Embedding等
    2. 将情感词库加入nn:
        1. 问题：如何将情感词汇对应的label信息加入进来？
    3. 动词、名词的subcategory
    4. Dependency relation
    5. 通过观察数据，考察虚词在两个子任务中起到的作用，再决定将虚词的哪些信息加入模型
    6. ...

2. 尝试其他模型结构：
    1. 直接使用Embedding作为分类特征（参考'Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms'一文）
    2. 使用Transformer最为sentence encoder（参见'Attention Is All You Need'一文)
3. task1/2分别调参
4. Error analysis: 总结错误中的pattern, 分析模型误判的可能原因、各类隐喻的分布及语言学特征
3. task1/2分别调参
5. 整理数据中的标注错误，联系组委询问
6. 拉其他成员？
7. Use F-score，macro F-score as eval metric

## Resources
1. Penn State中文隐喻语料库(http://www.personal.psu.edu/xxl13/download.html)
2. 大连理工情感词汇本体库(http://ir.dlut.edu.cn/EmotionOntologyDownload)

## Problems
* GPU?
