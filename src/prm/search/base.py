from vllm import LLM, SamplingParams

from src.prm.config import Config


PROMPT = """{}  
"""


def base(x, config: Config, llm: LLM):
    """
    x: dataset's elements
    """
    tokenizer = llm.get_tokenizer()
    convs = [
        [
            {"role": "user", "content": PROMPT.format()},
        ]
        for i in x
    ]
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
