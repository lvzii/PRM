import nlpertools


def calculate_overlap_rate(set1, set2):
    # 计算交集
    intersection = set1 & set2
    # 计算并集
    union = set1 | set2
    # 计算重合率
    overlap_rate = len(intersection) / len(union)
    return overlap_rate


def static_steps_num():
    # how much steps
    # 出错在哪步的概率
    all_step, wrong_step  = [], []
    baifenbi = []
    raw_data = nlpertools.load_from_json("../raw_data/false_cot.json")
    id2process = {f"{i['uid']}-{i['source']}": len(i['process']) for i in raw_data}
    for path in nlpertools.listdir("data_process-llama-3.1-instruct", including_dir=True):
        data= nlpertools.load_from_json(path)
        # all_step.append()
        # wrong_step.append()
        w_s = int(data["check_reponse"]["wrong_step"].lstrip("step "))
        a_s = id2process[nlpertools.get_filename(path, suffix=False)]
        baifenbi.append(w_s/a_s)
    print(baifenbi)
    data = baifenbi
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 使用matplotlib绘制直方图
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=5, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # 使用seaborn绘制密度图
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data, shade=True)
    plt.title('Density Plot of Data')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    plt.savefig("temp.png")
    from collections import Counter
    print(Counter(all_step))
    print(Counter(wrong_step))
    


def static():
    data1 = nlpertools.j_listdir("data_process-llama-3.1-instruct", including_dir=False)
    data2 = nlpertools.j_listdir("data_process-qwen_coder_2.5", including_dir=False)
    data1 = [i.split("-")[0] for i in data1]
    print(data1[:3])
    data2 = [i.split("-")[0] for i in data2]

    set1 = set(data1)
    set2 = set(data2)
    # print(len(set1), len(set2))

    overlap_rate = calculate_overlap_rate(set1, set2)
    print(f"重合率: {overlap_rate:.2f}")
    print(len(set1 & set2), len(set2))
    print(len(set1 & set2) / len(set2))

    print(len(set1 & set2), len(set1))
    print(len(set1 & set2) / len(set1))

    # 多出来的占所有的数量
    print(len(set2 - set1) / len(set1))

def temp():
    import matplotlib.pyplot as plt

    # 数据
    # true_count = {5: 5124, 6: 3124, 4: 2001, 7: 932, 8: 217, 3: 156, 9: 77, 10: 19, 11: 9, 2: 3, 15: 1, 13: 1, 12: 1}
    # false_count = {5: 4728, 6: 3827, 4: 1819, 7: 1654, 8: 499, 9: 165, 3: 121, 10: 52, 11: 22, 12: 9, 13: 7, 14: 4, 2: 2, 21: 1, 18: 1}
    
    true_count = {5: 884, 6: 671, 4: 326, 7: 263, 8: 74, 9: 34, 3: 22, 10: 8, 11: 3, 14: 1, 2: 1}
    false_count = {3: 928, 2: 785, 4: 274, 1: 139, 5: 122, 6: 27, 7: 11, 9: 1}
    max_num = max(list(true_count.keys()) + list(false_count.keys()))
    new_true_count = {i: true_count.get(i,0)  for i in range(1, max(list(true_count.keys()) + list(false_count.keys())) + 1)}
    new_false_count = {i: false_count.get(i,0)  for i in range(1, max(list(true_count.keys()) + list(false_count.keys())) + 1)}
    true_count = new_true_count
    false_count = new_false_count


    three_count = {i:false_count.get(i-3,0) for i in range(1, max_num + 1)}
    # 对键进行排序
    true_keys = sorted(true_count.keys())
    false_keys = sorted(false_count.keys())
    three_keys = sorted(three_count.keys())

    # 提取对应的值
    true_values = [true_count[key] for key in true_keys]
    false_values = [false_count[key] for key in false_keys]
    three_values = [three_count[key] for key in three_count]

    # 绘制折线图
    plt.figure(figsize=(10, 6))

    # True 分布
    # plt.plot(true_keys, true_values, marker='o', label='True', color='blue')

    # False 分布
    # plt.plot(false_keys, false_values, marker='o', label='False', color='red')

    # True 分布
    plt.plot(true_keys, true_values, marker='o', label='Total step', color='blue')

    # False 分布
    plt.plot(false_keys, false_values, marker='o', label='False step location', color='red')

    plt.plot(three_keys, three_values, marker='o', label='False step location(right shift)', color='pink')
    # 添加标题和标签
    # plt.title('True and False Distribution')
    plt.title('False step Distribution')
    plt.xlabel('step')
    plt.ylabel('Count')

    # 设置横坐标为整数
    plt.xticks(range(min(true_keys + false_keys), max(true_keys + false_keys) + 1))
    # plt.xticks(range(min(true_keys + false_keys + three_keys), max(true_keys + false_keys + three_keys) + 1))

    # 添加图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()
    # plt.savefig("temp_true_false.png")
    plt.savefig("temp_false_loc.png")
# static()
static_steps_num()

# temp()
