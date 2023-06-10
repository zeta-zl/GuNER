from transformers import AutoTokenizer, AutoModel

import torch
from torchcrf import CRF
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import time

import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall
import random
import numpy as np
import torch.nn.functional as F
import re
from tqdm.auto import tqdm

src_file_name = ""  # 源文件路径
tgt_file_name = ""  # 目标文件路径

model_path = ""

label_to_idx = {
    "[PAD]": 0, '[CLS]': 1,
    '[SEP]': 2,
    "B-PER": 3, "I-PER": 4,
    "B-OFI": 5, "I-OFI": 6,
    "B-BOOK": 7, "I-BOOK": 8,
    "O": 9
}
idx_to_label = {0: '[PAD]',
                1: '[CLS]', 2: '[SEP]',
                3: 'B-PER', 4: 'I-PER',
                5: 'B-OFI', 6: 'I-OFI',
                7: 'B-BOOK', 8: 'I-BOOK',
                9: 'O'}

padding = '[pad]'

config = {
    'fc_dim_1': 512,
    'lstm_out_1': 64,
    'lstm_out_2': 20,
    'lr': 0.001,
    'weight_decay': 5 * 1e-5,
    'batch_size': 32,
    'label_num': 10,
    'epochs': 1000,
    'save_dir': './model_',
}
device = 'cpu'


class NerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("Jihuai/bert-ancient-chinese")
        self.bert_layer = AutoModel.from_pretrained("Jihuai/bert-ancient-chinese")

        self.lstm_1 = nn.LSTM(768, config['lstm_out_1'], num_layers=1, bidirectional=True)
        self.dropout_1 = nn.Dropout(0.2)
        self.lstm_2 = nn.LSTM(2 * config['lstm_out_1'], config['lstm_out_2'], num_layers=1, bidirectional=False)
        self.dropout_2 = nn.Dropout()
        self.bn_1 = nn.BatchNorm1d(config['lstm_out_2'])
        self.linear_1 = nn.Linear(config['lstm_out_2'], config['label_num'])

        self.crf = CRF(config['label_num'])

    def fc(self, x):
        x, (hn, cn) = self.lstm_1(x)
        x = self.dropout_1(x)
        x, (hn, cn) = self.lstm_2(x)
        shape = x.shape
        x = x.view((-1, shape[-1]))

        x = self.bn_1(x)
        x = x.view(shape)
        x = self.linear_1(x)
        return x

    def _get_bert_feats(self, sentence):
        sentence_idx = self.tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        bert_output = self.bert_layer(**sentence_idx)
        return bert_output['last_hidden_state'].permute(1, 0, 2), (sentence_idx['attention_mask'] > 0).permute(1, 0)

    def cal_loss(self, sentence, tags):
        bert_feats, bert_mask = self._get_bert_feats(sentence)
        bert_feats = self.fc(bert_feats)
        crf_output = self.crf(bert_feats, tags.permute(1, 0), mask=bert_mask)
        return crf_output

    def forward(self, sentence):
        bert_feats, bert_mask = self._get_bert_feats(sentence)
        bert_feats = self.fc(bert_feats)
        return self.crf.decode(bert_feats, mask=bert_mask)


def load_model(path):
    _model = NerModel().to(device)
    _model.load_state_dict(torch.load(path))
    _model.eval()
    return _model


class mydataset(Dataset):
    def __init__(self, name):
        with open(name, 'r', encoding='utf-8') as f:
            temp = [i.strip() for i in f.readlines()]
            l = len(temp)
            self.text = temp[:l // 2]
            self.label = temp[l // 2:]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        _text = "".join(self.text[index].split())
        _label = [['[CLS]'] + self.label[index].split() + ['[SEP]']]
        _label = torch.tensor([label_to_idx[i] for i in _label[0]])
        return _text, _label


def collate_fn(batch_data):
    x, y = map(list, zip(*batch_data))
    y = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    return x, y


def get_data(file_name):
    train_set = mydataset(file_name)
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=1,
                                  shuffle=False,
                                  collate_fn=collate_fn)
    return train_dataloader


tags = [("B-PER", "I-PER"), ("B-OFI", "I-OFI"), ("B-BOOK", "I-BOOK")]


def _find_tag(labels, B_label="B-PER", I_label="I-PER"):
    result = []
    if isinstance(labels, str):
        labels = labels.strip().split()
        labels = ["O" if label == "O" else label for label in labels]
        # print(labels)
    song_pos0 = 0
    for num in range(len(labels)):
        if labels[num] == B_label:
            song_pos0 = num
        if labels[num] == I_label and labels[num - 1] == B_label:
            lenth = 2
            for num2 in range(num, len(labels)):
                if labels[num2] == I_label and labels[num2 - 1] == I_label:
                    lenth += 1
                if labels[num2] != I_label or num2 == len(labels) - 1:
                    result.append((song_pos0, lenth))
                    break
        if labels[num] != I_label and labels[num - 1] == B_label:
            result.append((song_pos0, 1))

    return result


def find_all_tag(labels):
    """
    :param labels:字符串形式的labels
    :return: 字典，key是各个label，value是一个(start,len)的元组
    """
    result = {}
    for tag in tags:
        res = _find_tag(labels, B_label=tag[0], I_label=tag[1])
        result[tag[0].split("-")[1]] = res
    return result


def label_to_txt(label, text):
    text = text[0]
    label = label.strip().split()
    dic = find_all_tag(label)
    items_ls = [(k, v) for k, v in dic.items()]
    _dic = {}
    for i in items_ls:
        for j in i[1]:
            _dic[j[0]] = (j, i[0])
    _dic_ls = [i for i in _dic.items()]
    _dic_ls.sort()
    ret_text = ""
    _in = False
    for i in range(len(text)):
        if label[i] == 'O':
            ret_text += text[i]
            _in = False
        elif label[i][0] == 'I':
            if _in:
                continue
            else:
                ret_text += text[i]
        else:
            ret_text += '{' + text[_dic[i][0][0]:_dic[i][0][0] + _dic[i][0][1]] + '|' + _dic[i][1] + '}'
            _in = True
    return ret_text


if __name__ == '__main__':
    model = load_model(model_path)
    dataloader = get_data(src_file_name)
    with open(tgt_file_name, 'w', encoding='utf-8') as f:
        for batch, (text, label) in enumerate(tqdm(dataloader)):

            pre = model(text)
            pre = [i if idx_to_label[i] != '[SEP]' else 9 for i in pre[0]]
            _pre = " ".join([idx_to_label[i] for i in pre[0][1:-1]])
            _pre_str = " ".join([idx_to_label[i] for i in pre[0][1:-1]])
            print(_pre_str)
            for i in _pre_str:
                if i == '[SEP]':
                    i = 'O'
            try:
                pre_text = label_to_txt(_pre_str, text)
            except:
                print(_pre_str, text)
                print(pre)
            f.write(pre_text)
            f.write('\n')
    print('done')
