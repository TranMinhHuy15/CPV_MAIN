import pandas as pd
import csv

def read_csv_to_list_pandas(filename):
    df = pd.read_csv(filename)
    return df.id1.tolist(),df.id2.tolist()

# 使用示例
id1,id2 = read_csv_to_list_pandas('score.csv')
id1=[str(id).zfill(5)for id in id1] 
id2=[str(id).zfill(5)for id in id2] 

li = []
for i, id in enumerate(id1):
    a=False
    for l in li:
        if id in l:
            if id2[i] not in l:
                l.append(id2[i])
            a=True
            continue
    if not a:
        li.append([id,id2[i]])
with open('a.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['Folder Name', 'JPG Count'])
        # 写入数据
        writer.writerows(li)