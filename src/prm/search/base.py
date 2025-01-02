from vllm import LLM, SamplingParams

from src.prm.config import Config
from src.prm.utils.api import aLLM


ANSWER_REQUIREMENT = """ """


def base(x, config: Config, llm: LLM):
    """
    x: dataset's elements
    """
    convs = [
        [
            {
                "role": "user",
                "content": i + ANSWER_REQUIREMENT,
            }
        ]
        for i in x["question"]
    ]
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


def base_api(x, config: Config, llm: aLLM):
    """
    dataset
    """
    responses = []
    for i in x["question"]:
        cur_response = []
        for _ in range(config.infer_times):
            response = llm.call(i + ANSWER_REQUIREMENT)
            print(response)
            cur_response.append(response)
        responses.append(cur_response)
    x["response"] = responses
    return x
