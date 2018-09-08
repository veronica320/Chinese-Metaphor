# Chinese-Metaphor
CCL 2018 Shared Task - 中文隐喻识别与情感分析

## Task Description
* 任务细节: http://ir.dlut.edu.cn/news/detail/508
* 训练数据: http://ir.dlut.edu.cn/File/Download?cid=3 《CCL 2018 中文隐喻识别与情感分析评测数据》
* 测试数据（无标签）: http://ir.dlut.edu.cn/File/Download?cid=3 《CCL 2018 中文隐喻识别与情感分析测试数据》

## Organizor
大连理工大学信息检索研究室
* Email: 林鸿飞 irlab@dlut.edu.cn
* QQ group: 604185567

## Code Structure

## Done
1. Baseline model: CGRU
2. Use F-score，macro F-score as eval metric
3. 获取中文隐喻语料库(http://www.personal.psu.edu/xxl13/download.html)，可用于自训练word embedding
4. 尝试以下feature：
4.1 Embedding层：
4.1.1 用Pre-trained word embedding代替模型自己学习的embedding
4.2 


## Todolist
1. 整理出数据中的标注错误，联系组委询问
2. Error analysis: 找出错误中的pattern
3. 拉其他成员？
4. 尝试更多feature:
5. 尝试其他模型结构：LSTM, attention...
6. 调参

## Problems
* GPU?
