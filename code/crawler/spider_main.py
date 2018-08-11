#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : spider_main.py
# @Author: smx
# @Date  : 2018/8/8
# @Desc  :

from syder.ispider import html_downloader
from syder.ispider import html_parser
from syder.ispider import html_outputer
from syder.ispider import url_manager
from syder.ispider.extract_words import extract_words
import os
class SpiderMain(object):
    def __init__(self):
        self.urls = url_manager.UrlManager()
        self.downloader = html_downloader.HtmlDownloader()
        self.parser = html_parser.HtmlParser()
        self.outputer = html_outputer.HtmlOutputer()
    def craw(self, file_path, root_url,word_num=5,count_num=100):
        key_words_dict = {}
        count = 1
        self.urls.add_new_url(root_url)

        if not os.path.exists(file_path):
            f = open(file_path, 'w')
        else:
            f = open(file_path, 'a')

        while self.urls.has_new_url():
            new_url = self.urls.get_new_url()
            print("craw %d : %s" % (count, new_url))
            # 爬取结果
            html_cont = self.downloader.download(new_url)
            new_urls, new_data = self.parser.paser(new_url, html_cont)

            # 将爬取结果抽取主题词
            if new_data != None:
                print('craw %d data:%s' %(count,new_data))
                key_words = extract_words(new_data['summary'], word_num=word_num)
                if key_words != None:
                    key_words_dict[new_data['title']] = key_words
                    f.write(new_data['title'] + ' ' + str(key_words) + '\n')

            # 添加新的url
            self.urls.add_new_urls(new_urls)
            self.outputer.collect_data(new_data)
            if count == count_num:
                break
            count = count + 1
        self.outputer.output_html()

        f.close()
        return key_words_dict

if __name__ == '__main__':
    root_url = 'https://baike.baidu.com/item/%E7%99%BE%E5%BA%A6/6699?fromtitle=baidu&fromid=107002'
    file_path = 'topic_words.txt'
    obj_spider = SpiderMain()
    key_words = obj_spider.craw(file_path=file_path,root_url=root_url,count_num=500)
    print(key_words)


