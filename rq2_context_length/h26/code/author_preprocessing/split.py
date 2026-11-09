import pandas as pd
import csv
import random


def read_csv_to_list_pandas(filename):
    df = pd.read_csv(filename)
    return df.id.tolist(),df.target.to_list()

# 使用示例
id, target = read_csv_to_list_pandas('train.csv')

id1=[str(i).zfill(5) for i, t in zip(id, target) if t==1]
id0=[str(i).zfill(5) for i, t in zip(id, target) if t==0]

id1=random.sample(id1, 150)
id0=random.sample(id0, 150)
id0.sort()
id1.sort()

data={}

for i in id1:
    data[i]=1
for i in id0:
    data[i]=0

print(data)
# my_dict = data

# # 按键排序（字母顺序）
# sorted_keys = sorted(my_dict.keys())
# for key in sorted_keys:
#     print(f"'{key}': {my_dict[key]}",end=',')

# 输出：
# apple: 2
# banana: 3
# orange: 4
# pear: 1

# id1=[str(id).zfill(5)for id in id1] 
# id2=[str(id).zfill(5)for id in id2] 

# li = []
# for i, id in enumerate(id1):
#     a=False
#     for l in li:
#         if id in l:
#             if id2[i] not in l:
#                 l.append(id2[i])
#             a=True
#             continue
#     if not a:
#         li.append([id,id2[i]])
# with open('a.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         # 写入表头
#         writer.writerow(['Folder Name', 'JPG Count'])
#         # 写入数据
#         writer.writerows(li)