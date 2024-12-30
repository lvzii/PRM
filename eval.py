import nlpertools
import argparse

# import torch
# from vllm import LLM

from src.prm.utils import score
from src.prm.utils.data import get_dataset, save_dataset
from src.prm.search import base, cot, rw
from src.prm.config import Config

STRATEGY_ALL = "all"
STRATEGY_BASE = "base"
STRATEGY_COT = "+cot"
STRATEGY_RW = "+rw"

TASK_PRF = "prf"
TASK_PASS = "pass"
TASK_THRESHOLD = "threshold"

config = Config()


def evaluate(args):
    dataset, task, strategy, gen_model, rw_model = args.dataset, args.task, args.strategy, args.gen_model, args.rw_model
    print(
        f"Args\ndataset: {args.dataset}\ntask: {args.task}\nstrategy: {args.strategy}\ngen_model: {args.gen_model}\nrw_model: {args.rw_model}\n\nEvaluation started...\n"
    )

    # llm = LLM(model=gen_model)
    llm = 1
    if rw_model:
        prm = 1
    dataset = get_dataset()

    infer_fn = {"base": base, "cot": cot, "rw": rw}[strategy]
    dataset = dataset.map(
        infer_fn,
        fn_kwargs={"llm": llm, "prm": prm},
    )
    dataset = score(dataset, config)
    save_dataset(dataset, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script with different strategies and tasks.")
    parser.add_argument("dataset", type=str, help="dataset path")
    parser.add_argument(
        "task", choices=[TASK_PRF, TASK_PASS, TASK_THRESHOLD], help="The task to execute (prf or downstream)"
    )
    parser.add_argument(
        "strategy",
        choices=[STRATEGY_BASE, STRATEGY_COT, STRATEGY_RW],
        help="The strategy to execute (base, +cot, or advanced)",
    )

    parser.add_argument("gen_model", type=str, help="gen model path")
    parser.add_argument("rw_model", type=str, help="rw model path")

    args = parser.parse_args()

    evaluate(args)
    # python  eval.py  syntax pass base / //
