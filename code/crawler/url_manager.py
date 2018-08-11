#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : url_manager.py
# @Author: smx
# @Date  : 2018/8/8
# @Desc  : RUL管理器，管理要下载的网页和将要下载的网页url，将待爬去的url传给网页下载器

class UrlManager(object):
    def __init__(self):
        self.new_urls = set()
        self.old_urls = set()

    def add_new_url(self, url):
        if url is None:
            return
        if url not in self.new_urls and url not in self.old_urls:
            self.new_urls.add(url)

    def add_new_urls(self, urls):
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)

    def has_new_url(self):
        return len(self.new_urls) != 0

    def get_new_url(self):
        new_url = self.new_urls.pop()
        self.old_urls.add(new_url)
        return new_url
