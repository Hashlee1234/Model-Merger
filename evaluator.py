import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from typing import List
import json


DEFAULT_TEST_CASES = [
    {
        "prompt": "What is the capital of France?",
        "reference": "The capital of France is Paris."
    },
    {
        "prompt": "Explain what a neural network is in one sentence.",
        "reference": "A neural network is a computational system inspired by the human brain that learns patterns from data."
    },
    {
        "prompt": "What is 2 + 2?",
        "reference": "2 + 2 equals 4."
    },
    {
        "prompt": "Write a Python function to add two numbers.",
        "reference": "def add(a, b):\n    return a + b"
    },
    {
        "prompt": "What is the boiling point of water?",
        "reference": "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit."
    },
]


class ModelEvaluator:

    def __init__(self, tokenizer: AutoTokenizer, max_new_tokens: int = 100):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def generate(self, model, prompt: str) -> str:
        model.eval()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def score_rouge(self, generated: str, reference: str) -> dict:
        scores = self.rouge.score(reference, generated)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    def score_bertscore(self, generated: List[str], references: List[str]) -> dict:
        P, R, F1 = bert_score(
            generated,
            references,
            lang="en",
            verbose=False,
            device="cpu"
        )
        return {
            "bert_precision": round(P.mean().item(), 4),
            "bert_recall":    round(R.mean().item(), 4),
            "bert_f1":        round(F1.mean().item(), 4),
        }

    def evaluate_model(self, model, model_name: str, test_cases: list = None) -> dict:
        if test_cases is None:
            test_cases = DEFAULT_TEST_CASES

        print(f"\n[Evaluator] Evaluating: {model_name}")

        results = {
            "model_name": model_name,
            "per_prompt": [],
            "aggregate": {}
        }

        all_generated  = []
        all_references = []
        all_rouge1     = []
        all_rouge2     = []
        all_rougeL     = []

        for i, tc in enumerate(test_cases):
            prompt    = tc["prompt"]
            reference = tc["reference"]

            print(f"  [{i+1}/{len(test_cases)}] Prompt: {prompt[:50]}...")

            generated = self.generate(model, prompt)
            print(f"           Output : {generated[:80]}...")

            rouge_scores = self.score_rouge(generated, reference)

            results["per_prompt"].append({
                "prompt":    prompt,
                "reference": reference,
                "generated": generated,
                **rouge_scores
            })

            all_generated.append(generated)
            all_references.append(reference)
            all_rouge1.append(rouge_scores["rouge1"])
            all_rouge2.append(rouge_scores["rouge2"])
            all_rougeL.append(rouge_scores["rougeL"])

        print(f"  Computing BERTScore...")
        bert_scores = self.score_bertscore(all_generated, all_references)

        results["aggregate"] = {
            "avg_rouge1": round(sum(all_rouge1) / len(all_rouge1), 4),
            "avg_rouge2": round(sum(all_rouge2) / len(all_rouge2), 4),
            "avg_rougeL": round(sum(all_rougeL) / len(all_rougeL), 4),
            **bert_scores
        }

        print(f"  Results: ROUGE-1={results['aggregate']['avg_rouge1']} | "
              f"ROUGE-L={results['aggregate']['avg_rougeL']} | "
              f"BERTScore-F1={results['aggregate']['bert_f1']}")

        return results

    def compare_models(self, models_dict: dict, test_cases: list = None) -> dict:
        all_results = {}

        for name, model in models_dict.items():
            all_results[name] = self.evaluate_model(model, name, test_cases)

        winner = max(
            all_results.keys(),
            key=lambda k: all_results[k]["aggregate"]["bert_f1"]
        )

        return {
            "results":     all_results,
            "winner":      winner,
            "metric_used": "bert_f1"
        }


if __name__ == "__main__":
    print("Evaluator module loaded. Import and use ModelEvaluator class.")
    print("See app.py for full usage example.")
