from transformers import AutoTokenizer, AutoModel

import torch
from torchcrf import CRF
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import time
import numpy as np

anno = '''
src_file_name：数据文件
saved_model_dir：可选，用于加载训练的模型
NerModel：模型描述
mask_token：随机遮盖部分输入
train：核心代码
其它函数为辅助函数，或是评分函数
直接运行即可
'''

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
src_file_name = ""  # 数据文件
valid_file_name = ""
config = {
    'fc_dim_1': 512,
    'lstm_out_1': 64,
    'lstm_out_2': 20,
    'lr': 0.001,
    'weight_decay': 5 * 1e-5,
    'batch_size': 32,
    'label_num': 10,
    'epochs': 200,
    'save_dir': './model_',
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_dir = './log.txt'
saved_model_dir = ""


def write_log(s):
    if isinstance(s, list):
        s = " ".join([str(i) for i in s])
    if not isinstance(s, str):
        s = str(s)
    with open(log_dir, 'a+', encoding='utf-8') as f:
        f.writelines(str(s))
        f.write('\n')


write_log("train_start")


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


def split_train_test(dataset, r=0.1):
    l = dataset.__len__()
    train_set_size = int(l * (1 - r))
    test_set_size = l - train_set_size
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_set_size, test_set_size],
        generator=torch.Generator().manual_seed(0)
    )
    return train_dataset, test_dataset


def get_data(file_name):
    train_set, test_set = split_train_test(mydataset(file_name))
    train_set_l, test_set_l = train_set.__len__(), test_set.__len__()
    train_dataset = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True,
                               collate_fn=collate_fn)
    test_dataset = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate_fn)
    return train_dataset, test_dataset, train_set_l, test_set_l, train_set, test_set


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


def mask_token(sentence, rate=0.2):
    mask_num = int(rate * len(sentence))
    mask_array = np.zeros(len(sentence))
    mask_array[:mask_num] = 1
    np.random.shuffle(mask_array)
    new_sentence = ''
    for i in range(len(sentence)):
        if mask_array[i] == 1:
            new_sentence += '[MASK]'
        else:
            new_sentence += sentence[i]
    return new_sentence


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
                if labels[num2] == "O":
                    result.append((song_pos0, lenth))
                    break
    return result


def find_all_tag(labels):
    result = {}
    for tag in tags:
        res = _find_tag(labels, B_label=tag[0], I_label=tag[1])
        result[tag[0].split("-")[1]] = res
    return result


def _precision(pre_labels, true_labels):
    '''
    :param pre_tags: list
    :param true_tags: list
    :return:
    '''
    pre = []
    if isinstance(pre_labels, str):
        pre_labels = pre_labels.strip().split()
        pre_labels = ["O" if label == "O" else label for label in pre_labels]
    if isinstance(true_labels, str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label == "O" else label for label in true_labels]

    pre_result = find_all_tag(pre_labels)
    for name in pre_result:
        for x in pre_result[name]:
            if x:
                if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                    pre.append(1)
                else:
                    pre.append(0)
    return sum(pre), len(pre)


def _recall(pre_labels, true_labels):
    '''
    :param pre_tags: list
    :param true_tags: list
    :return:
    '''
    recall = []
    if isinstance(pre_labels, str):
        pre_labels = pre_labels.strip().split()
        pre_labels = ["O" if label == "O" else label for label in pre_labels]
    if isinstance(true_labels, str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label == "O" else label for label in true_labels]

    true_result = find_all_tag(true_labels)
    for name in true_result:
        for x in true_result[name]:
            if x:
                if pre_labels[x[0]:x[0] + x[1]] == true_labels[x[0]:x[0] + x[1]]:
                    recall.append(1)
                else:
                    recall.append(0)
    return sum(recall), len(recall)


def _get_all(pre_labels, true_labels):
    if isinstance(pre_labels, str):
        pre_labels = pre_labels.strip().split()
        pre_labels = ["O" if label == "O" else label for label in pre_labels]
    if isinstance(true_labels, str):
        true_labels = true_labels.strip().split()
        true_labels = ["O" if label == "O" else label for label in true_labels]
    pre_pairs = find_all_tag(pre_labels)
    true_pairs = find_all_tag(true_labels)

    pre_ls = [t for v in pre_pairs.values() for t in v]
    true_ls = [t for v in true_pairs.values() for t in v]
    correct_ls = len(set(pre_ls) & set(true_ls))
    return correct_ls, len(pre_ls), len(true_ls)


def _f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0


def train():
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.
        train_batch_loss_ls = []
        test_batch_loss_ls = []
        for batch, (text, label) in enumerate((train_dataloader)):
            try:
                for i in range(len(text)):
                    text[i] = mask_token(text[i], 0.4)

                optimizer.zero_grad()
                label = label.to(device)
                loss = (-1) * model.cal_loss(text, label).to(device)
                loss.backward()
                optimizer.step()
                lr_ls.append(optimizer.state_dict()['param_groups'][0]['lr'])
                lr_scheduler.step()
                running_loss += float(loss)
            except ValueError:
                print()
                print(text, label)
                print()
        print("epoch:", epoch + 1, "loss:", running_loss / train_set_l, end=" ")
        write_log(["epoch:", epoch + 1, "loss:", running_loss / train_set_l])
        train_loss_ls.append(running_loss / train_set_l)

        model.eval()
        running_loss = 0.
        for batch, (text, label) in enumerate((test_dataloader)):
            try:
                label = label.to(device)
                loss = (-1) * model.cal_loss(text, label).to(device)
                running_loss += float(loss)

            except ValueError:
                print()
                print(text, label)
                print()
        print("epoch:", epoch + 1, "valid loss:", running_loss / test_set_l, end=" \n")
        write_log(["valid loss:", running_loss / test_set_l])
        valid_loss_ls.append(running_loss / test_set_l)
        print("   lr:", optimizer.state_dict()['param_groups'][0]['lr'])
        if (epoch + 1) % 10 == 0:
            train_f1_ls.append(_valid_cal_f1("", model, train_set)[-1])
            valid_f1_ls.append(_valid_cal_f1("", model, test_set)[-1])
            print("train f1: ", train_f1_ls[-1], "test f1:", valid_f1_ls[-1])

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       config['save_dir'] + "_".join(time.ctime()[4:-5].replace(":", "_").split(" ")) + str(
                           epoch) + ".model")


def load_model(path):
    _model = NerModel().to(device)
    _model.load_state_dict(torch.load(path))
    _model.eval()
    return _model


def _valid_cal_f1(path, model=None, dataset=None):
    if model is None:
        model = load_model(path)
    if dataset is None:
        valid_set = mydataset(valid_file_name)
    else:
        valid_set = dataset
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    res_ls = []
    label_ls = []
    cor_ls, all_ls, pos_ls = [], [], []
    model.eval()
    for text, label in (valid_loader):
        pre = model(text)
        res_ls += pre[0][1:-1]
        label_ls += label.tolist()[0][1:-1]
        _res = " ".join([idx_to_label[i] for i in pre[0][1:-1]])
        _label = " ".join([idx_to_label[i] for i in label.tolist()[0][1:-1]])
        cor, all, pos = _get_all(_res, _label)
        cor_ls.append(cor)
        all_ls.append(all)
        pos_ls.append(pos)

    cor_sum = sum(cor_ls)
    all_sum = sum(all_ls)
    pos_sum = sum(pos_ls)
    _pre = cor_sum / all_sum
    _rec = cor_sum / pos_sum
    f1 = _f1_score(_pre, _rec)
    return _pre, _rec, f1


if __name__ == '__main__':

    train_dataloader, test_dataloader, train_set_l, test_set_l, train_set, test_set = get_data(src_file_name)

    if True:
        try:
            model = load_model(saved_model_dir)
            print("from_save")
        except:
            model = NerModel().to(device)
            print("new")

    for name, para in model.named_parameters():
        if "bert_layer" in name:
            para.requires_grad_(False)

    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "CRF" not in n], "lr": config['lr']},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "CRF" in n],
         "lr": config['lr'] * 200},
    ]
    optimizer = optim.Adam(params, weight_decay=config['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config['lr'] / 100, max_lr=config['lr'],
                                                     step_size_up=int(
                                                         config['epochs'] * train_set_l / config['batch_size'] / 40),
                                                     cycle_momentum=False, gamma=0.95)

    write_log(model)
    train_loss_ls = []
    valid_loss_ls = []
    train_f1_ls = []
    valid_f1_ls = []
    lr_ls = []

    train()
    write_log(str(train_loss_ls))
    write_log(str(valid_loss_ls))
    write_log(str(train_f1_ls))
    write_log(str(valid_f1_ls))

    valid_file_name = ""
    _pre, _rec, f1 = _valid_cal_f1("", model)

    write_log(str((_pre, _rec, f1)))
