"""
evaluation.py
--------------------------------------
ROUGE Evaluation for Text Summarization

Requires:
pip install rouge-score
"""

from rouge_score import rouge_scorer
from tabulate import tabulate


class RougeEvaluation:

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def evaluate(self, reference, generated):
        """
        Returns ROUGE scores.
        """

        scores = self.scorer.score(reference, generated)

        result = {
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-L": scores["rougeL"].fmeasure
        }

        return result

    def compare(self,
                reference,
                extractive_summary,
                abstractive_summary):
        """
        Compare Extractive and Abstractive summaries.
        """

        ext = self.evaluate(reference, extractive_summary)
        abs_sum = self.evaluate(reference, abstractive_summary)

        table = [
            [
                "Extractive",
                round(ext["ROUGE-1"], 3),
                round(ext["ROUGE-2"], 3),
                round(ext["ROUGE-L"], 3)
            ],
            [
                "Abstractive",
                round(abs_sum["ROUGE-1"], 3),
                round(abs_sum["ROUGE-2"], 3),
                round(abs_sum["ROUGE-L"], 3)
            ]
        ]

        print("\n")
        print("=" * 60)
        print("ROUGE SCORE COMPARISON")
        print("=" * 60)

        print(
            tabulate(
                table,
                headers=[
                    "Method",
                    "ROUGE-1",
                    "ROUGE-2",
                    "ROUGE-L"
                ],
                tablefmt="grid"
            )
        )

        print("=" * 60)

        return ext, abs_sum


def manual_comparison():

    print("\n")
    print("=" * 60)
    print("MANUAL COMPARISON")
    print("=" * 60)

    print("""
Extractive Summary
------------------
✔ More factual
✔ Uses original sentences
✔ No new words generated
✔ Better for preserving information

Abstractive Summary
-------------------
✔ More human-like
✔ Easier to read
✔ Can generate new sentences
✔ Better fluency
✔ Better compression

Conclusion
----------
Extractive is better for accuracy.
Abstractive is better for readability.
""")