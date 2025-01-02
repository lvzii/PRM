from vllm import LLM, SamplingParams

from src.prm.config import Config


PROMPT = """{question}

Please answer me with the json format :{"CoT": "Tell me  how to fix the bug use the chain of thought, step by step", "buggy_code": "The buggy code in the systemverilog (just one line of code)", "correct_code": "The correct code (just one line of code that can directly replace the buggy code, without any other description)"}"""


def cot(x, config: Config, llm: LLM):
    """
    x: dataset's elements
    """
    convs = [
        [
            {"role": "user", "content": PROMPT.format(question=i)},
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


def cot_api():
    pass
