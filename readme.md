
目录树：
```
├─数据处理                      
│      preprocess.py      预处理      
│      sep.py             预处理后重整格式 
│      split.py           随机划分训练集和验证集
│
├─模型
│  └─bert-ancient-chinese_overfitting_mask
│          re_ft.py       重微调，具体使用方式将在技术报告中给出
│          train.py       训练代码
│
└─输出
        submit.py         输出结果

```