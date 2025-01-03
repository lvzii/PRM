import json
import os
import nlpertools
import concurrent.futures
from tqdm import tqdm
import time

from prm.utils.api import aLLM
from prm.utils.parse import process_response
from eval_utils import get_prompt


def _get_dataset_path():
    if DATASET == "syntax":
        return "./dataset/verilog-machine-syntax-test.json"
    elif DATASET == "function":
        return "./dataset/verilog-machine-function-test.json"
    else:
        raise ValueError("DATASET must be syntax or function")


def func(idx, i):
    response = []
    save_path = os.path.join(ROOT_DIR, f"{idx}.json")
    if os.path.exists(save_path):
        output = nlpertools.load_from_json(save_path)
        response = output["response"]
    for each in response:
        # 去除之前失败的
        if each:
            response.append(each)
    if len(response) == GENERATE_TIMES:
        return output
    else:
        generate_times = GENERATE_TIMES - len(response)

    prompt = get_prompt(DATASET, STRATEGY, i)
    for _ in range(generate_times):
        response_content = allm.call(prompt, temperature=TEMPERATURE)
        response_content = process_response(response_content)
        if response_content:
            response_json = json.loads(response_content)
            response.append(response_json)
    i["response"] = response
    nlpertools.save_to_json(i, save_path)
    return i


if __name__ == "__main__":
    DATASET = "syntax"
    STRATEGY = "cot"
    TEMPERATURE = 0.1
    MAX_WORKERS = 64
    GENERATE_TIMES = 20
    ROOT_DIR = f"eval_result/res-{DATASET}-{STRATEGY}"
    allm = aLLM(api_port=8001)
    nlpertools.j_mkdir(ROOT_DIR)

    while 1:
        for GENERATE_TIMES in range(1, 21):
            dataset_path = _get_dataset_path()
            data = nlpertools.load_from_json(dataset_path)
            new_data = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(func, idx, i) for idx, i in enumerate(data)]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc="Processing"):
                    new_data.append(future.result())

            nlpertools.save_to_json(new_data, f"{ROOT_DIR}.json")

            print("over")
            time.sleep(1)
