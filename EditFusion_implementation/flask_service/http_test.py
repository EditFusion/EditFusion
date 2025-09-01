import requests

# 目标URL
url = "http://localhost:5002/es_predict"

params = {
    "base": "a\nb\nc\nd\ne\n",
    "ours": "a\nb\nc\nd\n111\ne\n",
    "theirs": "333\na\n222\nc\nd\ne\n444\n",
}
response = requests.get(url, params=params)

if response.status_code == 200:
    # 打印响应内容
    print(response.json()["data"])
else:
    # 打印错误信息
    print("Failed to retrieve data:", response.status_code)
