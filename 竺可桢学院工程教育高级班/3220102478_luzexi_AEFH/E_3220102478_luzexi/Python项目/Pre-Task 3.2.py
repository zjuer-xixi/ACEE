# -*- codeing = utf-8 -*-
# @Time : 2023/3/20 19:34
import re
from bs4 import BeautifulSoup    #导入需要的库

file = open("./bilibili.html","rb")   #打开html文件
html = file.read()
soup = BeautifulSoup(html, 'html.parser')
result = soup.find_all("a",href=re.compile(r'^https://www\.bilibili\.com/video/BV[\w\d]+'))#运用find_all方法查找
for item in result:
    print(item['href'])       #输出
