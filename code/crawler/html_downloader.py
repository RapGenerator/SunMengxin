#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : html_downloader.py
# @Author: smx
# @Date  : 2018/8/8
# @Desc  :下载从url_manager收到的网页内容，保存成一个字符串交给网页解析器解析

import urllib.request
class HtmlDownloader(object):
    def download(self, url):
        if url is None:
            return None
        response = urllib.request.urlopen(url,timeout=10)
        if response.getcode() != 200:
            return None
        return response.read()
