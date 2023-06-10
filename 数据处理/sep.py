src_file_name = ""   # 源文件路径
tgt_file_name = ""  # 目标文件路径

if __name__ == '__main__':
    with open(src_file_name, 'r', encoding='utf-8') as f:
        with open(tgt_file_name, 'w', encoding='utf-8') as f1:
            ls = f.readlines()
            for i in range(0, len(ls), 2):
                f1.write(ls[i])
            for i in range(1, len(ls), 2):
                f1.write(ls[i])
