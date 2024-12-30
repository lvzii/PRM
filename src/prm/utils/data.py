import datasets
from datasets import Dataset, load_dataset

QUESTION_PROMPT = """Help me fix bug in verilog/systemverilog.
<spec>
{spec}
</spec>

Here is the code that is supposed to keep corresponding to the spec, but there is a bug in it:
<bug_code>
{bug_code}
</bug_code>"""

SYNTAX_PROMPT = """Help me fix bug in verilog/systemverilog.
<spec>
{spec}
</spec>

Here is the code that is supposed to keep corresponding to the spec, but there is a bug in it:
<bug_code>
{bug_code}
</bug_code>

Here is iverilog's compilation error message:
{syntax_log}"""


def _syntax(i):
    prompt = QUESTION_PROMPT.format(spec=i["spec"], bug_code=i["buggy_code"], syntax_log=i["syntax_log"])
    i["question"] = prompt
    return i


def _function(i):
    i["question"] = QUESTION_PROMPT.format(spec=i["spec"], bug_code=i["buggy_code"])
    return i


def get_dataset(dataset_name):
    prompn_fn = {"syntax": _syntax, "function": _function}[dataset_name]
    data_file = {"syntax": "verilog-machine-syntax-test.json", "function": "verilog-machine-test.json"}[dataset_name]
    dataset = load_dataset(
        "json",
        data_files=data_file,
    )
    # print(dataset.)
    dataset = dataset.map(prompn_fn, batched=False)
    return dataset


def save_dataset(dataset, config):
    dataset.to_json(f"{config.output_dir}/completions.json")
