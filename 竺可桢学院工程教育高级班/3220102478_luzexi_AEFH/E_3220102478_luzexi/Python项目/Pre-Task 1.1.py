#导入requests库
import requests
#使用get请求
html = requests.get("http://www.4399.com/")
#判断请求是否成功
assert html.status_code == 200
#确定编码方式
html.encoding = "utf-8"
#显示结果
print(html.text)

