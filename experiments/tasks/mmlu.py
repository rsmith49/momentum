import re

import pandas as pd

from experiments.core import BaseTask

SYSTEM_PROMPT = """
You are a well rounded professional who is an expert in multiple domains. Answer the questions asked of you and enclose the final answer in [brackets].
"""
QUESTION_PROMPT_TEMPLATE = "Question: {question}\nPossible Choices: {choices}"


class MMLU(BaseTask):
    task_name = "mmlu"

    def create_user_question(self, row: pd.Series) -> str:
        return QUESTION_PROMPT_TEMPLATE.format(
            question=row["question"],
            choices=row["choices"],
        )

    def create_system_prompt(self, row: pd.Series) -> str:
        return SYSTEM_PROMPT

    def is_llm_correct(self, row: pd.Series, llm_completion: str) -> bool:
        def clean(s: str) -> str:
            s = s.lower().strip()
            symbols_to_clear = [".", "$", "#", "@", "(", ")", "&", "!", "?", ";", '"', "'", "`", "’", "‘"]
            if len(s) > 0 and s[-1] in symbols_to_clear:
                s = s[:-1]
                s = clean(s)
            if len(s) > 0 and s[0] in symbols_to_clear:
                s = s[1:]
                s = clean(s)

            return s

        true_answer = row["choices"][row["answer"]]
        # Fairly naive string matching
        matches = re.findall(r"\[([^\[\]]*)]", llm_completion)
        if len(matches) > 0:
            return clean(matches[-1]) == clean(true_answer)

        return False
