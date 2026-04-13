# LLM Architecture Inspector

I built this because I kept hearing terms like "attention heads", "hidden size", and "parameter count" without really understanding what they meant visually. So I made a tool that lets you actually see inside any language model 

## What it does

You give it a model path, and it shows you:

- Every single layer in the model, nested inside its parent (exactly like a DOM tree)
- The shape of every weight matrix — so you can see that q_proj is [2560, 2560] and understand why it's called a square projection
- How many parameters each part holds, and how much memory it takes in MB
- A chart showing which layers are the "heaviest" in terms of parameter count

I tested it on Microsoft's Phi-2 (2.7B parameters) and it correctly mapped out all 32 decoder layers, 193 linear layers, and the full attention mechanism.

## Why I built it this way

Most LLM tutorials just say "the model has X billion parameters" without showing you where those parameters live. I wanted to actually see the structure — which is why I modeled it after a DOM tree. Once I made that connection (every layer is like an HTML element with children), the whole thing clicked.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-architecture-inspector
cd llm-architecture-inspector
pip install -r requirements.txt
streamlit run app.py
```

You'll need a model downloaded locally. I used Phi-2 from HuggingFace:
- Download from: https://huggingface.co/microsoft/phi-2
- Files needed: `config.json`, `tokenizer.json`, `tokenizer_config.json`, both `.safetensors` files
- Then point the app to your local folder path in the sidebar

## How to use it

1. Run `streamlit run app.py`
2. In the sidebar, paste your local model path )
3. Click **Inspect Model**
4. Explore the 4 tabs: Summary, DOM Tree, Parameter Shapes, Charts

The DOM Tree tab is the interesting one — you can see exactly how `model.layers.0.self_attn.q_proj` fits inside `self_attn`, which fits inside `PhiDecoderLayer`, all the way up to the root.

## Tech used

- `transformers` for model loading
- `torch` for the inspection APIs (`named_modules`, `named_parameters`, `state_dict`)
- `streamlit` for the UI
- `plotly` for the parameter distribution charts


Built this as part of teaching myself LLM internals. Happy to answer questions about how any part of it works.
