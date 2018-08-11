# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from S2S.Seq2Seq.data_helpers import *
from S2S.Seq2Seq.model import Seq2SeqModel
import math

if __name__ == '__main__':

    # 超参数
    rnn_size = 1024
    num_layers = 2
    embedding_size = 1024
    batch_size = 128
    learning_rate = 0.0001
    epochs = 200 #迭代次数=数据使用次数
    filepath = 'data/data.txt'

    # 加载并预处理数据
    data = load_data(filepath)
    processed_data, word_to_id, _ = process_all_data(data)

    # 加载模型，分配GPU和CPU，用with开头调用session才能作为下文默认的session，with语句结果之后close session，
    with tf.Session() as sess:
        # 1.获取模型的对象
        model = Seq2SeqModel(rnn_size, num_layers, embedding_size, learning_rate, word_to_id,
                             mode='train', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)
        #2. 初始化节点
        # tf.global_variables_initializer()：初始化的变量节点，可在训练完成后保存在磁盘上，之后可重新加载
        # tensorflow的中的变量时内存缓存区中的变量，需要显示初始化
        # 构建完成整个模型之后，运行这个节点
        sess.run(tf.global_variables_initializer())#模型开始运行节点进行初始化操作

        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            batches = getBatches(processed_data, batch_size)
            for nextBatch in batches:
                loss, summary = model.train(sess, nextBatch) #计算模型的损失函数和梯度下降的值
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')#语言模型中评判预测结果的指标，PPL越小，模型越好
                print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
                model.saver.save(sess, 'model/seq2seq.ckpt')