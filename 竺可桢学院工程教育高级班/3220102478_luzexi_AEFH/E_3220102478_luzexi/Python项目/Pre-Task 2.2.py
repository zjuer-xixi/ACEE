# -*- codeing = utf-8 -*-
# @Time : 2023/3/21 16:47
import re

def number(s):
    judge = r"-?\d+"  # 匹配整数或带负号的整数
    number = re.findall(judge, s)    # 使用re库找出数字
    return [int(a) for a in number]
print(number('abc123j120c0-1'))  #此处也可填入其他字符
