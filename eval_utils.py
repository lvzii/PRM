QUESTION_PROMPT = """Help me fix bug in verilog/systemverilog.
<spec>
{spec}
</spec>

Here is the code that is supposed to keep corresponding to the spec, but there is a bug in it:
<bug_code>
{bug_code}
</bug_code>

"""

SYNTAX_PROMPT = """Help me fix bug in verilog/systemverilog.
<spec>
{spec}
</spec>

Here is the code that is supposed to keep corresponding to the spec, but there is a bug in it:
<bug_code>
{bug_code}
</bug_code>

Here is iverilog's compilation error message:
<compile_log>
{syntax_log}
</compile_log>

"""

ANSWER_REQUIREMENT_BASE = 'Please answer me with the json format :{"bug_line": "The buggy code in the systemverilog (just one line of code)", "fixed_line": "The correct code (just one line of code that can directly replace the buggy code, without any other description)"}'
ANSWER_REQUIREMENT_COT = 'Please answer me with the json format :{"CoT": "Tell me  how to fix the bug use the chain of thought, step by step", "bug_line": "The buggy code in the systemverilog (just one line of code)", "fixed_line": "The correct code (just one line of code that can directly replace the buggy code, without any other description)"}'


def get_prompt(dataset, strategy, i):
    if dataset == "syntax":
        question = SYNTAX_PROMPT.format(spec=i["spec"], bug_code=i["buggy_code"], syntax_log=i["syntax_log"])
    elif dataset == "function":
        question = QUESTION_PROMPT.format(spec=i["spec"], bug_code=i["buggy_code"])
    else:
        raise ValueError("DATASET must be syntax or function")
    if strategy == "base":
        return question + ANSWER_REQUIREMENT_BASE
    elif strategy == "cot":
        return question + ANSWER_REQUIREMENT_COT
    else:
        raise ValueError("STRATEGY must be base or cot")
