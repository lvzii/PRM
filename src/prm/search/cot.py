from vllm import LLM, SamplingParams
import concurrent.futures
from tqdm import tqdm
from prm.utils.parse import process_response
from src.prm.config import Config
from src.prm.utils.api import aLLM


ANSWER_REQUIREMENT = 'Please answer me with the json format :{"CoT": "Tell me  how to fix the bug use the chain of thought, step by step", "bug_line": "The buggy code in the systemverilog (just one line of code)", "fixed_line": "The correct code (just one line of code that can directly replace the buggy code, without any other description)"}'


def cot(x, config: Config, llm: LLM):
    """
    x: dataset's elements
    """
    convs = [
        [
            {"role": "user", "content": i + ANSWER_REQUIREMENT},
        ]
        for i in x["question"]
    ]

    # 删除question列
    x = x.remove_columns("question")
    tokenizer = llm.get_tokenizer()
    templated_convs = tokenizer.apply_chat_template(
        convs,
        tokenize=False,
    )
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )
    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    x["response"] = responses

    return x


def cot_api(x, config: Config, llm: aLLM):
    """
    dataset
    because of multi-processing conflict with dataset.map, so we should implement multi-processing in this function
    """

    # todo
    def func(idx, i):
        cur_response = []
        for _ in range(config.infer_times):
            try:
                response = llm.call(i + ANSWER_REQUIREMENT)
            except Exception as e:
                continue
            # here to process response
            response = process_response(response)
            if response is None:
                continue
            # print(response)
            cur_response.append(response)

        return idx, cur_response

    data = x["question"]
    responses = [None] * len(data)
    # multi-processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.api_max_workers) as executor:
        futures = [executor.submit(func, idx, i) for idx, i in enumerate(data)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc="Call LLM Processing"):
            idx, response = future.result()
            responses[idx] = response
    # single-processing
    # for i in x["question"]:
    #     cur_response = []
    #     for _ in range(config.infer_times):
    #         response = llm.call(i + ANSWER_REQUIREMENT)
    #         print(response)
    #         cur_response.append(response)
    #     responses.append(cur_response)
    del x["question"]
    x["response"] = responses
    return x

    # same time.
    # data = [i for i in x["question"] * config.infer_times]
