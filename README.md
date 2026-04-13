# LLM Model Merger 

Merge two large language models into one using three state-of-the-art strategies — **SLERP**, **TIES**, and **DARE** — with a full evaluation pipeline comparing merged vs original models.

## What It Does

- **3 merging strategies** implemented in pure PyTorch
- **Full evaluation pipeline** — ROUGE + BERTScore + LLM output comparison  
- **Streamlit UI** — visual controls for blend factor, strategy params, results charts
- Works entirely on **CPU** — no GPU required
- Compatible with any HuggingFace `AutoModelForCausalLM` model


## Project Structure

```
llm_merger/
├── merger_core.py   
├── evaluator.py     
├── app.py           
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
