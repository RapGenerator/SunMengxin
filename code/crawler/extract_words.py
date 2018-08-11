#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_words.py
# @Author: smx
# @Date  : 2018/8/8
# @Desc  : 从一段文字中抽取关键字

import jieba
import jieba.analyse
import re

def extract_words(text,word_num=5):
    # 第一步：分词，这里使用结巴分词全模式
    fenci_text = jieba.cut(text)

    # 第二步：去停用词
    stopwords = {}.fromkeys([line.rstrip() for line in open('stopwords.txt',mode='r',encoding='utf-8')])

    final = ""
    for word in fenci_text:
        if word not in stopwords:
            if (word != "。" and word != "，"):
                final = final + " " + word
    # print(final)

    # 第三步：提取关键词
    ex_words = jieba.analyse.extract_tags(text, topK=word_num, withWeight=True, allowPOS=())
    ex_words = [e[0] for e in ex_words]
    # # 第四步：去除分开的数字
    # pattern = re.compile(r'[1-9]\d*')
    # num_index = [i for i,e in enumerate(ex_words) if pattern.match(e)]
    #
    # if len(num_index) == 0:
    #     words = [e[0] for e in ex_words]
    # else:
    #     stops = [' ',',','.','。',';','\:','\"','\'','[',']','{','}','(',')','!','~','`','@','#','$','%','^','&','*',]
    #     for i in num_index:
    #         inx = text.find(ex_words[i])
    #         now_word = text[inx:inx+len(ex_words[i])]
    #         pre_word = text[max(inx-1,0)]
    #         after_word = text[min(len(text),inx+len(ex_words[i]))]
    #         if now_word.find(pre_word) == -1 and now_word.find(after_word) == -1:
    #             if pre_word not in stops:
    #                 ex_words[i] = pre_word + ex_words[i]
    #                 if inx - 2 > 0:
    #                     ex_words[i] = text[inx-2] + ex_words[i]  # 取前面的两个词
    #                 continue
    #
    #             if after_word not in stops:
    #                 ex_words[i] = ex_words[i] + after_word
    #                 if inx+len(ex_words[i])+1 < len(text):
    #                     ex_words[i] = ex_words[i] + text[inx+len(ex_words[i])]  # 取后面的两个词

    return ex_words

if __name__ == '__main__':
    text = 'Python2.7是纯粹的自由软件， ' \
           '源代码和解释器Python2.7CPython遵循 GPL（GNU General Public License）许可。' \
           'Python2.7语法简洁清晰，特色之一是强制用空白符（white space）作为语句缩进。'

    words = extract_words(text)

    print(words)

