from dataclasses import dataclass


@dataclass
class Config:
    # model params

    output_dir: str = "output"
    infer_times: int = 20

    ## strategy rw params
    rw_max_branch_num: int = 5
