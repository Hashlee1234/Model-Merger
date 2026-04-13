# LLM Model Merger ⚗️

Merge two large language models into one using three state-of-the-art strategies — **SLERP**, **TIES**, and **DARE** — with a full evaluation pipeline comparing merged vs original models.

> Built as part of an LLM systems engineering portfolio.

## What It Does

- **3 merging strategies** implemented from scratch in pure PyTorch
- **Full evaluation pipeline** — ROUGE + BERTScore + LLM output comparison  
- **Streamlit UI** — visual controls for blend factor, strategy params, results charts
- Works entirely on **CPU** — no GPU required
- Compatible with any HuggingFace `AutoModelForCausalLM` model

## Merging Strategies

| Strategy | How it works | Best for |
|---|---|---|
| **SLERP** | Spherical interpolation between weight vectors | Smooth capability blending |
| **TIES** | Trim noise → elect sign → merge agreements | Models fine-tuned on different tasks |
| **DARE** | Drop 90% of deltas, rescale rest, then merge | Reducing interference between models |

## Project Structure

```
llm_merger/
├── merger_core.py   # Core engine: SLERP, TIES, DARE implementations
├── evaluator.py     # ROUGE + BERTScore evaluation pipeline
├── app.py           # Streamlit web UI
├── requirements.txt
└── README.md
```

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/llm-model-merger
cd llm-model-merger

pip install -r requirements.txt

streamlit run app.py
```

## Usage

1. Download any two HuggingFace models locally
2. Set paths in the sidebar
3. Choose strategy + blend factor
4. Click **Merge Models**
5. Click **Evaluate & Compare** to see how merged performs vs originals

## Key Concepts

**Task vector** = `fine_tuned_weights - base_weights`  
The delta that represents what a model learned during fine-tuning.  
All 3 strategies operate on task vectors, not raw weights.

**SLERP** travels the spherical arc between vectors rather than a straight line,  
preserving the "direction" (semantic meaning) of weight space.

**TIES** resolves the sign conflict problem — when model A says a weight should  
increase and model B says decrease, naive averaging cancels both out. TIES votes.

**DARE** exploits the finding that ~90% of delta weights are redundant noise  
and can be dropped without hurting performance, making merges cleaner.

## Author

Built by [YOUR NAME] — LLM systems engineering portfolio project.
