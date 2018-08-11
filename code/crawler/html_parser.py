#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : html_parser.py
# @Author: smx
# @Date  : 2018/8/8
# @Desc  : 解析出有价值的数据，解析出url交给url管理器

from bs4 import BeautifulSoup
import re
import urllib.parse


class HtmlParser(object):
    def _get_new_urls(self, page_url, soup):
        new_urls = set()
        links = soup.find_all('a', href=re.compile(r'/item/'))
        for link in links:
            new_url = link['href']
            new_full_url = urllib.parse.urljoin(page_url, new_url)
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_data(self, page_url, soup):

        if soup == None:
            return None

        res_data = {}
        # url
        res_data['url'] = page_url

        try:
            title_node = soup.find('dd', class_="lemmaWgt-lemmaTitle-title").find('h1')
        except Exception:
            print('soup has no find')
            return None

        res_data['title'] = title_node.get_text()

        # lemma-summary
        summary_node = soup.find('div', class_="lemma-summary")
        if summary_node == None:
            res_data['summary'] = 'None'
        else:
            res_data['summary'] = summary_node.get_text()

        return res_data

    def paser(self, page_url, html_cont):
        if page_url == None or html_cont == None:
            return

        soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
        new_urls = self._get_new_urls(page_url, soup)
        new_data = self._get_new_data(page_url, soup)
        return new_urls, new_data
