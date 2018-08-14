#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : process_data.py
# @Author: smx
# @Date  : 2018/8/12
# @Desc  : 处理爬取后的数据

import os
import re
import csv

res_folder_path = 'res_data/process_res'
rap_folder_path = 'res_data/说唱'
album_files = os.listdir(rap_folder_path)
s = []

pattern = re.compile(r'([\d{1,2}:\d{1,2}:\d{1,2}])')

non_words = open('drop_words/nonwords.txt','r',encoding='utf-8').readlines()
censor_words = open('drop_words/censorwords.txt','r',encoding='utf-8').readlines()

for album_file in album_files: #所有专辑文件夹名称
    out = open(res_folder_path + '/' + album_file+'.csv', 'a', newline='',encoding='utf-8')
    csv_write = csv.writer(out, dialect='excel')

    files = os.listdir(rap_folder_path + '/' + album_file)
    for file in files:
        csv_write.writerow([' '])
        if not os.path.isdir(file):
            f = open(rap_folder_path + '/' + album_file + '/' +file,'r', encoding='UTF-8')
            lines = f.readlines()
            flag = 0
            for line in lines:
                # 不要时间戳
                out = re.sub(pattern,'',line)

                # 不要歌词开头介绍
                if out.find('：')>-1 or out.find(':') > -1:
                    continue

                # 不要歌词开头介绍
                if sum([1 for each in non_words if out.find(each.strip())>-1]) >0:
                    continue

                # 不要含有脏词的语句
                if sum([1 for each in censor_words if out.find(each.strip()) > -1]) > 0:
                    continue

                #不要空行
                out = re.sub('[+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）❤”“x×`~{}()<>.*&![]', "",out.strip())
                out = out.replace('Live', '').replace('live', '').replace('[','').replace(']','')
                if len(out) < 1:
                    continue
                else:
                    # 文件首行是歌曲名称
                    if flag == 0:
                        flag = 1
                        csv_write.writerow([file.replace('.txt','').replace('❤','')])
                    csv_write.writerow([out])
