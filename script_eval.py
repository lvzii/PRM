import nlpertools
import argparse


from src.prm.utils import score
from src.prm.utils.data import get_dataset, save_dataset
from src.prm.search import base, cot, rw, base_batch_api, cot_api, rw_api
from src.prm.config import Config

STRATEGY_ALL = "all"
STRATEGY_BASE = "base"
STRATEGY_COT = "cot"
STRATEGY_RW = "rw"

TASK_PRF = "prf"
TASK_PASS = "pass"
TASK_THRESHOLD = "threshold"

config = Config()


def load_llm(config):
    if config.engine == "vllm":
        import torch
        from vllm import LLM

        llm = LLM(model=config.gen_model)
    else:
        from src.prm.utils.api import aLLM

        llm = aLLM(api_port=config.api_port)
    return llm


def evaluate(args):
    dataset, task, strategy, gen_model, rw_model = args.dataset, args.task, args.strategy, args.gen_model, args.rw_model
    print(
        f"Args\ndataset: {args.dataset}\ntask: {args.task}\nstrategy: {args.strategy}\ngen_model: {args.gen_model}\nrw_model: {args.rw_model}\n\nEvaluation started...\n"
    )
    # combine args & config
    config.dataset = dataset
    config.task = task
    config.strategy = strategy
    config.gen_model = gen_model
    config.rw_model = rw_model

    llm = load_llm(config)
    dataset = get_dataset(dataset, config)
    infer_fn = {
        "api": {"base": base_batch_api, "cot": cot_api, "rw": rw_api},
        "vllm": {"base": base, "cot": cot, "rw": rw},
    }[config.engine][strategy]
    if config.strategy != STRATEGY_RW:
        # network package conflict with multi-processing in dataset.map, so api mode can't set num_proc > 1
        dataset = dataset.map(
            infer_fn,
            fn_kwargs={"config": config, "llm": llm},
            batched=True,
            batch_size=20,
            num_proc=1,
            desc="Dataset Map INFER",
        )
    else:
        prm = 1
        dataset = dataset.map(infer_fn, fn_kwargs={"config": config, "llm": llm, "prm": prm}, batched=True)

    # dataset = score(dataset, config)
    save_dataset(dataset, config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluation script with different strategies and tasks.")
    parser.add_argument("--dataset", type=str, help="dataset path", default="syntax")
    parser.add_argument(
        "--task",
        choices=[TASK_PRF, TASK_PASS, TASK_THRESHOLD],
        help="The task to execute (prf or downstream)",
        default="pass",
    )
    parser.add_argument(
        "--strategy",
        choices=[STRATEGY_BASE, STRATEGY_COT, STRATEGY_RW],
        help="The strategy to execute (base, +cot, or advanced)",
        default=STRATEGY_COT,
    )

    parser.add_argument(
        "--gen_model", type=str, help="gen model path", default="/data/jiyoushu/hub/Qwen/Qwen2___5-Coder-7B-Instruct/"
    )
    parser.add_argument(
        "--rw_model", type=str, help="rw model path", default="/data/jiyoushu/hub/Qwen/Qwen2___5-Coder-7B-Instruct/"
    )

    args = parser.parse_args()

    evaluate(args)
    # CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py function pass base /data/jiyoushu/hub/Qwen/Qwen2___5-Coder-7B-Instruct/ /data/jiyoushu/hub/Qwen/Qwen2___5-Coder-7B-Instruct/
