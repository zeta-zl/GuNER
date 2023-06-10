import random

# 随机分割训练集和测试集
r = [i for i in range(2347)]
random.shuffle(r)

valid = r[:20]
# print(valid)

fm = open("GuNER2023_train.txt", "r", encoding="utf-8")

NER_valid = open("GuNER2023_train.valid.txt", "w", encoding="utf-8")

NER_train = open("GuNER2023_train.train.txt", "w", encoding="utf-8")

cnt = 0
while True:
    line = fm.readline()
    if line:
        if cnt in valid:
            NER_valid.write(line)
        else:
            NER_train.write(line)
    else:
        break
    cnt += 1

fm.close()
NER_valid.close()
NER_train.close()
