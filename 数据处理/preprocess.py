import re

src_file_name = ""  # 源文件路径
tgt_file_name = ""  # 目标文件路径
normal_tag = 'O'  # 标注无特殊意义的词语

'''
示例：
原文本：
帝曰：「{玄齡|PER}、{如晦|PER}不以勳舊進，特其才可與治天下者，{師合|PER}欲以此離間吾君臣邪？」斥嶺表。
想得奉飛蓋，曳長裾，藉玳筵，躡珠履，歌山桂之偃蹇，賦池竹之檀欒。其崇貴也如彼，其風流也如此，幸甚幸甚，何樂如之！高視上京，有懷{德祖|PER}，才謝天人，多慚{子建|PER}，書不盡意，寧俟繁辭。
处理后：
帝 曰 ： 「 玄 齡 、 如 晦 不 以 勳 舊 進 ， 特 其 才 可 與 治 天 下 者 ， 師 合 欲 以 此 離 間 吾 君 臣 邪 ？ 」 斥 嶺 表 。
O O O O B-PER I-PER O B-PER I-PER O O O O O O O O O O O O O O O O B-PER I-PER O O O O O O O O O O O O O O O
想 得 奉 飛 蓋 ， 曳 長 裾 ， 藉 玳 筵 ， 躡 珠 履 ， 歌 山 桂 之 偃 蹇 ， 賦 池 竹 之 檀 欒 。 其 崇 貴 也 如 彼 ， 其 風 流 也 如 此 ， 幸 甚 幸 甚 ， 何 樂 如 之 ！ 高 視 上 京 ， 有 懷 德 祖 ， 才 謝 天 人 ， 多 慚 子 建 ， 書 不 盡 意 ， 寧 俟 繁 辭 。
O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-PER I-PER O O O O O O O O B-PER I-PER O O O O O O O O O O O

'''

tags_ls = ["P", "O", "B"]  # "PER","OFI","BOOK"
dic = {
    "P": "PER",
    "O": "OFI",
    "B": "BOOK"
}

if __name__ == '__main__':
    with open(src_file_name, 'r', encoding='utf-8') as f:
        with open(tgt_file_name, 'w', encoding='utf-8') as f1:
            pattern = r'\{.*?\}'  # 正则匹配{}中
            ls = f.readlines()
            for i in ls:  # 对每一句进行处理
                i = i.strip()  # 去除'\n'


                def repl(match):
                    # 接受一个匹配对象，替换成实体类型
                    k, v = match.group()[1:-1].split("|")
                    # dic[k[0]] = "B-"+v
                    return v[0] * len(k)


                def _repl(match):
                    # 接受一个匹配对象，返回|前的实体名
                    k, v = match.group()[1:-1].split("|")
                    return k


                temp = re.sub(pattern, repl, i)
                _temp = re.sub(pattern, _repl, i)

                tags = [normal_tag] * len(temp)
                pre = normal_tag
                for pos, item in enumerate(temp):
                    if item in tags_ls:
                        v = dic[item]
                        if pre == v:
                            tags[pos] = "I-" + v
                        else:
                            pre = v
                            tags[pos] = "B-" + v
                    else:
                        pre = normal_tag
                        tags[pos] = normal_tag

                _data = " ".join(_temp)  # 输出原句经过正则替换后的句子
                _tags = " ".join(tags)  # 输出原句中，每个词对应的tag

                f1.write(_data)
                f1.write('\n')
                f1.write(_tags)
                f1.write('\n')
