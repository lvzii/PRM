import random
import sys
import nlpertools
from openai import OpenAI
import nlpertools
import re
import boto3
import nlpertools
import json
from tqdm import tqdm
from openai import OpenAI
from langchain.prompts import PromptTemplate
import threading
import concurrent.futures
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from pathlib import Path
import os
import pandas as pd
from itertools import groupby
from operator import itemgetter
from collections import Counter
from dataclasses import asdict, dataclass
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.dataprocess import compare_golden


@dataclass
class Process:
    step: str
    label: str  # True False Neural Unknown


# LABEL_PROCESS_LABEL_PROMPT = """Here is a spec in EDA:
# <spec>
# {spec}
# </spec>

# Here is the code that is supposed to keep corresponding to the spec, but there is a bug in it:
# <bug_code>
# {bug_code}
# </bug_code>

# Here is the thought process on how to analyze and fix bugs.
# <steps>
# {steps}
# </steps>

# Please reflect on this thought process(steps) and tell me where it went wrong.
# Inform me of the incorrect step, and provide me with the correct thought process and the correct fix result(It's fine to do it within the JSON field; there's no need to write it outside the JSON field.).
# The result should be returned in JSON format:
# {{
#     "wrong_step": "where did it go wrong? such as 'step 1' or 'step 2' or ...",
#     "wrong_reason": "why this step go wrong",
#     "true_cot": "true cot: step 1: xxx step 2: xxx ...",
#     "bug_line": "The buggy code in the systemverilog (just one line of code)",
#     "fixed_line": "The correct code (just one line of code that can directly replace the buggy code, without any other description)",
# }}"""

LABEL_PROCESS_LABEL_PROMPT = """Here is a spec in EDA: 
<spec>
{spec}
</spec>

Here is the code that is supposed to keep corresponding to the spec, but there is a bug in it:
<bug_code>
{bug_code}
</bug_code>

Here is the thought process on how to analyze and fix bugs.
<steps>
{steps}
</steps>

I will tell you where bug is in the buggy code for reference:
<bug_line>
{bug_line}
</bug_line>

Please reflect on this thought process(steps) and tell me where it went wrong. 
Inform me of the incorrect step, and provide me with the correct thought process and the correct fix result(It's fine to do it within the JSON field; there's no need to write it outside the JSON field.).
The result should be returned in JSON format:
{{
    "wrong_step": "where did it go wrong? such as 'step 1' or 'step 2' or ...",
    "wrong_reason": "why this step go wrong",
    "true_cot": "true cot: step 1: xxx step 2: xxx ...",
    "bug_line": "The buggy code in the systemverilog (just one line of code)",
    "fixed_line": "The correct code (just one line of code that can directly replace the buggy code, without any other description)",
}}"""


RW_TRAIN_PROMPT = """The following is a question and a thought step for answering, please judge whether the thought step is correct.
<question>
{question}
</question>

<chain_of_thought>
{cot}
</chain_of_thought>

Answer me if this chain of thought is correct(True/False)"""


def get_model_response(instruction):
    messages = [{"role": "user", "content": instruction}]
    response = client.chat.completions.create(messages=messages, model=MODEL, temperature=TEMPERATURE)
    return response.choices[0].message


def construct_check_prompt(i):
    question = i["question"]
    # ?AttributeError: 'dict' object has no attribute 'strip'
    spec = re.findall("<spec>(.*)</spec>", question, re.S)[0].strip()
    bug_code = re.findall("<bug_code>(.*)</bug_code>", question, re.S)[0].strip()
    steps = []
    process = i["process"]
    for sdx, s in enumerate(process):
        steps.append(f"step {sdx + 1}: {s["step"]}")
    prompt = LABEL_PROCESS_LABEL_PROMPT.format(
        spec=spec, bug_code=bug_code, steps="\n".join(steps), bug_line=i["golden_answer"]["bug_line"]
    )
    return prompt


def parse(response):
    if MODEL.startswith("qwen"):
        data = response.strip().lstrip("```json").rstrip("```").strip()
        try:
            data = json.loads(data)
            return data
        except Exception as e:
            print(data)
            print(e.__str__())
            return None
    elif MODEL.startswith("llama"):
        try:
            left = response.find("{")
            right = response.rfind("}")
            data = response[left : right + 1]
            data = json.loads(data)
            return data
        except Exception as e:
            # print(data)
            # print(e.__str__())
            return None
    else:
        raise "must be qwen/llama"


def parse_infer_result(v, i):
    """
    模型推理结果 ，再检查
    :return:
    """

    golden_answer = i["golden_answer"]
    if compare_golden(v["bug_line"], v["fixed_line"], golden_answer["bug_line"], golden_answer["fixed_line"]):
        return True
    else:
        return False


def modify_process(v, i):

    process = i["process"]
    new_process = []
    wrong_step = int(v["wrong_step"].lstrip("step "))
    for edx, each in enumerate(process):
        if edx + 1 != wrong_step:
            new_process.append(asdict(Process(step=each["step"], label="True")))
        else:
            new_process.append(asdict(Process(step=each["step"], label="False")))
            break
    return new_process


class OneDataFlow(object):
    def __init__(self):
        pass


def label():
    def func(i):
        save_path = f"{UNIT_RESPONSE_DIR}/{i["uid"]}-{i["source"]}.json"
        # step 0 filter already success
        if Path(save_path).exists():
            # print(f"pass {i["uid"]}-{i["source"]}")
            return
        # step 1 construct check_prompt, from `false cot`
        check_prompt = construct_check_prompt(i)
        for _ in range(MAX_TRY_TIMES):
            # step 2 call llm
            try:
                response = get_model_response(check_prompt).content
            except Exception as e:
                print(e.__str__())
            # step 2 check llm response
            response = parse(response)
            if not response:
                continue
            # step 3 check whether llm response is correct
            try:  # json is json,but term is not correct
                ok = parse_infer_result(response, i)
            except Exception as e:
                continue
            if not ok:
                continue
            # step 4 modify new_process
            response["source"] = MODEL
            i["check_reponse"] = response
            try:
                i["process"] = modify_process(response, i)
            except Exception as e:
                print(e.__str__())
                continue
            # step 5 save all info for debug
            nlpertools.save_to_json(i, save_path)
            print(f"success {i["uid"]}-{i["source"]}")
            break

    nlpertools.j_mkdir(UNIT_RESPONSE_DIR)
    data = nlpertools.load_from_json(RAW_DATA)
    # for i in data:
    #     func(i)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(func, i) for idx, i in enumerate(data)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc="Processing"):
            future.result()


def build_train_data():
    # combine all data
    data = []
    # Strategy 1:  Union two model result
    id_set = set()
    for dir_path in ["data_process-llama-3.1-instruct", "data_process-qwen_coder_2.5"]:
        for path in nlpertools.listdir(dir_path):
            # Strategy 1:  Union two model result
            if nlpertools.get_filename(path, suffix=False) in id_set:
                continue
            id_set.add(nlpertools.get_filename(path, suffix=False))
            i = nlpertools.load_from_json(path)
            question = i["question"]
            process = i["process"]
            cur_cot = []
            for edx, each in enumerate(process):
                cur_cot.append(each["step"])
                data.append(
                    {
                        "instruction": RW_TRAIN_PROMPT.format(question=question, cot="\n".join(cur_cot)),
                        "input": "",
                        "output": str(each["label"]),
                        "source": dir_path,
                    }
                )
    # shuffle
    random.shuffle(data)
    # balance true & false label num
    true_data = [i for i in data if i["output"] == "True"]
    false_data = [i for i in data if i["output"] == "False"]
    true_num = len(true_data)
    false_num = len(false_data)
    print("true_num:", true_num)
    print("false_num:", false_num)
    if true_num > false_num:
        true_data = true_data[:false_num]
    else:
        false_data = false_data[:true_num]
    data = true_data + false_data

    nlpertools.save_to_json(data, f"data_process-all.json")

    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

    # split temp_data into 1:1
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # print result
    print("trainset size:", len(train_data))
    print("valset size:", len(val_data))
    print("testset size:", len(test_data))
    nlpertools.save_to_json(train_data, f"rw-v7-train.json")
    nlpertools.save_to_json(test_data, f"rw-v7-test.json")
    nlpertools.save_to_json(val_data, f"rw-v7-val.json")


def back_data_real_time():
    data = []
    for dir_path in ["data_process-llama-3.1-instruct", "data_process-qwen_coder_2.5"]:
        for path in nlpertools.listdir(dir_path):
            cur = nlpertools.load_from_json(path)
            data.append({"data": cur, "source": path})
    nlpertools.save_to_json(data, "data_process-real-time.json")


if __name__ == "__main__":
    # API_PORT = "8001"
    MODEL = "llama-3.1-instruct"
    TEMPERATURE = 1.8
    API_PORT = "8000"
    # MODEL = "qwen_coder_2.5"

    client = OpenAI(
        base_url=f"http://localhost:{API_PORT}/v1",
        api_key="0",
    )
    MAX_TRY_TIMES = 5
    MAX_WORKERS = 64
    UNIT_RESPONSE_DIR = f"data_process-{MODEL}"
    RAW_DATA = "../raw_data/false_cot.json"

    # label()
    # back_data_real_time()
    build_train_data()
