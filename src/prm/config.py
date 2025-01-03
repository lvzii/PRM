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
    max_tokens = 16384
    top_p = 1.0
    # other
    num_samples: int = None  # 10
    output_dir: str = "output"
    infer_times: int = 20
    max_infer_times: int = 100  #

    ## strategy rw params
    rw_max_branch_num: int = 5

    # api
    api_port: int = 8001
    api_max_workers: int = 64
