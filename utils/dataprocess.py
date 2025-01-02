# encoding=utf-8
import random
import re

import nlpertools
import nltk


def split_train_test(train_triple_data, test_size=0.2, random_state=7, source_term="source"):
    """
    根据source字段将数据划分为训练集和测试集。

    :param train_triple_data: 包含所有数据的列表
    :param test_size: 测试集的比例，默认为0.2
    :param random_state: 随机种子，用于可重复性
    :return: 训练集和测试集

    # 示例用法
    train_triple_data = [
        {"instruction": "指令1", "input": "输入1", "chosen": "选择1", "rejected": "拒绝1", "source": 1},
        {"instruction": "指令2", "input": "输入2", "chosen": "选择2", "rejected": "拒绝2", "source": 2},
        {"instruction": "指令3", "input": "输入3", "chosen": "选择3", "rejected": "拒绝3", "source": 1},
        {"instruction": "指令4", "input": "输入4", "chosen": "选择4", "rejected": "拒绝4", "source": 3},
    ]

    train_set, test_set = split_train_test(train_triple_data, test_size=0.5, random_state=42)

    print("训练集:", train_set)
    print("测试集:", test_set)
    """
    if random_state is not None:
        random.seed(random_state)

    # 获取所有唯一的source
    sources = list(set(item[source_term] for item in train_triple_data))

    # 随机打乱sources
    random.shuffle(sources)

    # 计算测试集的大小
    num_test_sources = int(len(sources) * test_size)

    # 划分source为训练集和测试集
    test_sources = sources[:num_test_sources]
    train_sources = sources[num_test_sources:]

    # 根据source划分数据
    train_set = [item for item in train_triple_data if item[source_term] in train_sources]
    test_set = [item for item in train_triple_data if item[source_term] in test_sources]

    return train_set, test_set


class Filter:
    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            r"C:\Users\23702\Desktop\Projects\EDA\models\deepseek-vl-1.3b-base"
        )

    def filter(self, text):
        if len(text) > 30000:
            return False
        if len(self.tokenizer.tokenize(text)) > 4096:
            return False
        if text.count("always") >= 3:
            return True
        return False


def get_module_name(code):
    # code = nlpertools.readtxt_string(r"C:\Users\23702\Desktop\Projects\EDA\task\DataManage\inserted_assert_from_shaokai_sv_passed_syntax\a23_coprocessor_7459_8_1_1.sv")
    res = re.findall("^module[\s]*(.*?)[\s\n(]", code)
    if not res:
        return "?"
    # print(res)
    return res[0].strip()


def extract_assert_from_code(code):
    res = re.findall("(property .*)endmodule", code, re.S)
    return res[0].strip()


def del_assert(code):
    former = code.split("// Assertion Start ***************************")[0]
    return former + "\nendmodule"


# 去除字符串中的所有空格并进行预处理
def preprocess_for_compare_semtic_consist(s):
    # 去除注释
    s = re.sub(r"//.*$", "", s)
    # 去除所有空格
    s = re.sub(r"\s+", "", s)

    # 移除行末标点符号
    s = s.rstrip(";,")
    # 处理 input 和 output 声明中的默认 wire 类型
    s = re.sub(r"\b(input|output)\s+wire\b", r"\1", s)

    return s


def compare_golden(response_bug_line, response_fixed_line, golden_bug_line, golden_fixed_line):
    try:
        # try的是none
        if preprocess_for_compare_semtic_consist(response_fixed_line) == preprocess_for_compare_semtic_consist(
            golden_fixed_line
        ) and preprocess_for_compare_semtic_consist(response_bug_line) == preprocess_for_compare_semtic_consist(
            golden_bug_line
        ):
            return True
        else:
            return False
    except:
        return False


def del_annotation(code):
    lines = code.split("\n")
    in_multiline_comment = False

    new_lines = []
    for line in lines:
        raw_line = line
        # 去除行首和行尾的空白字符
        line = line.strip()

        if not line:
            continue  # 跳过空行

        if in_multiline_comment:
            # 检查多行注释是否结束
            if "*/" in line:
                in_multiline_comment = False
                line = line.split("*/", 1)[1]
            else:
                continue  # 跳过多行注释的行

        # 检查单行注释
        if line.startswith("//"):
            continue  # 跳过单行注释

        # 检查多行注释开始
        if "/*" in line:
            if "*/" in line:
                # 多行注释在同一行内结束
                line = line.split("/*", 1)[0] + line.split("*/", 1)[1]
            else:
                in_multiline_comment = True
                line = line.split("/*", 1)[0]

        # 去除行尾的单行注释
        line = re.split(r"//", line, 1)[0].strip()

        if line:
            # line_count += 1
            new_lines.append(raw_line)
    return "\n".join(new_lines)


if __name__ == "__main__":
    get_module_name(code="")
