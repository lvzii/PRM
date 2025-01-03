import json

from utils.dataprocess import compare_golden
from nlpertools import estimate_pass_at_k


def func(x):
    correct_num = 0
    responses = x["response"]
    for response in responses:
        response = json.loads(response)
        res = compare_golden(
            response["bug_line"],
            response["fixed_line"],
            x["golden_answer"]["bug_line"],
            x["golden_answer"]["fixed_line"],
        )
        if res:
            correct_num += 1
    x["correct_num"] = correct_num
    x["response_num"] = len(responses)
    return x


def score(dataset, config):
    dataset = dataset.map(func, batched=False, desc="Scoring")

    sample_num, correct_num = [], []
    for i in dataset:
        sample_num.append(i["response_num"])
        correct_num.append(i["correct_num"])
    print(sample_num)
    print(correct_num)
    print("pass@1", round(estimate_pass_at_k(sample_num, correct_num, 1).mean() * 100, 2))
    print("pass@5", round(estimate_pass_at_k(sample_num, correct_num, 5).mean() * 100, 2))
    print()
