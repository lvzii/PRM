import nlpertools
import nltk
import re


def split_sentence4en(text):
    # 该方法，适用于claude3.5等英文推理cot的分句
    # 对codellama不行，数字的位置会被放到前一行
    sentences = nltk.sent_tokenize(text)
    # nlpertools.print_split()

    # 首先如果是\n\n数字，先切分。
    new_sentences = []
    for idx, i in enumerate(sentences):
        if re.match(".*\n\n\d+.", i, re.S):
            sp = i.split("\n\n")
            if len(sp) == 2:
                new_sentences.extend(sp)
            else:
                a, b = "\n\n".join(sp[:-1]), sp[-1]
                new_sentences.extend([a, b])
        else:
            new_sentences.append(i)
    # 合并数字
    sentences = new_sentences
    new_sentences = []
    pre = ""
    for idx, i in enumerate(sentences):
        if idx != 0 and re.match("^\d+\.", pre):
            new_sentences[-1] = new_sentences[-1] + i
        else:
            new_sentences.append(i)
        pre = i
    # for i in new_sentences:
    #     nlpertools.print_split(sign="-")
        # print(i)
    sentences = new_sentences
    return sentences
