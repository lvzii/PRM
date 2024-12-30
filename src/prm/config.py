from dataclasses import dataclass

from typing import Literal


@dataclass
class Config:
    # base setting
    task: Literal["prf", "pass", "threshold"] = "pass"
    strategy: Literal["base", "+cot", "+rw", "all"] = "base"
    dataset: Literal["syntax", "function"] = "function"
    gen_model: str = "/data/jiyoushu/hub/Qwen/Qwen2___5-Coder-7B-Instruct/"
    rw_model: str = None

    engine: Literal["vllm", "api"] = "api"
    # model params

    temperature = 1.0
    max_tokens = 8192
    top_p = 1.0
    # other

    output_dir: str = "output"
    infer_times: int = 20

    ## strategy rw params
    rw_max_branch_num: int = 5
