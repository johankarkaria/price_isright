# The Price Is Right

An autonomous multi-agent system that scans the web for product deals, estimates each item's true market value using an ensemble of pricing models, and pushes a notification when it finds a genuine bargain.

## How it works

```
Scanner Agent  →  Ensemble Agent  →  Messaging Agent
(finds deals)     (estimates value)    (sends alert)
```

1. **Scanner Agent** pulls listings from deal RSS feeds and uses an LLM with structured outputs to select the 5 best-described, clearly-priced deals.
2. **Ensemble Agent** estimates each item's true value with three independent pricing models, combined by a weighted average:
   - **Frontier Agent** — retrieves similar priced products from a Chroma vector store and asks a frontier LLM to estimate price with that context (RAG)
   - **Specialist Agent** — a fine-tuned open-source LLM served on Modal GPU infrastructure (see below)
   - **Neural Network Agent** — a deep residual feed-forward network trained on hashed text features
3. **Planning Agent** coordinates the above, and once a deal's discount clears a threshold, hands it to the **Messaging Agent**, which crafts a summary and sends a Pushover push notification.

An **Autonomous Planning Agent** variant replaces the fixed pipeline with an LLM that decides which tools to call and when, using function calling.

A Gradio dashboard (`price_is_right.py`) runs the framework continuously, showing live agent logs, discovered deals, and a 3D t-SNE plot of the product vector store.

## Fine-tuned pricing model

The Specialist Agent is backed by a custom fine-tuned model rather than a prompted API call:

- **Base model:** `meta-llama/Llama-3.2-3B`
- **Fine-tuning method:** QLoRA — the base model is loaded in 4-bit (`BitsAndBytesConfig`, NF4) and a **PEFT/LoRA** adapter is trained on top for the pricing task
- **Task format:** `"What does this cost to the nearest dollar?\n\n{description}\n\nPrice is $..."`
- **Distribution:** the trained adapter is pushed to the Hugging Face Hub and loaded at a pinned commit revision, so inference is reproducible
- **Serving:** deployed serverlessly on **Modal** on a T4 GPU, either as a stateless function (`pricer_ephemeral.py`) or a warm, class-based service with a cached model volume (`pricer_service2.py`)

## Project structure

```
agents/
  agent.py                     Base class for colored, named agent logging
  scanner_agent.py              Scrapes and filters deals from RSS feeds
  frontier_agent.py             RAG-based pricing via vector search + LLM
  specialist_agent.py           Calls the fine-tuned model on Modal
  neural_network_agent.py       Calls the deep neural network pricer
  deep_neural_network.py        Model definition and inference for the NN pricer
  ensemble_agent.py             Combines the three pricing models
  preprocessor.py                Rewrites raw descriptions into a clean format
  planning_agent.py             Fixed scan → price → notify pipeline
  autonomous_planning_agent.py  Tool-calling agent version of the pipeline
  messaging_agent.py            Pushover notifications, Claude-crafted summaries
  deals.py                      Deal/Opportunity data models and RSS scraping
  items.py                      Training data model, HuggingFace Hub I/O

deal_agent_framework.py        Orchestrator: memory, vector store, plotting
price_is_right.py              Gradio UI and live dashboard
evaluator.py                   Scatter-plot/error evaluation for pricing models
pricer_service.py / pricer_service2.py / pricer_ephemeral.py
                                Modal deployments of the fine-tuned pricer
hello.py, llama.py             Minimal Modal examples
```

## Setup

```bash
git clone https://github.com/johankarkaria/price_isright.git
cd price_isright
uv sync   # or: pip install -r requirements.txt
```

Create a `.env` file with the required credentials: `OPENAI_API_KEY`, `PUSHOVER_USER` / `PUSHOVER_TOKEN`, plus any keys for the LLMs referenced (e.g. Anthropic, for the message-crafting step).

The Specialist Agent expects a fine-tuned pricer deployed on [Modal](https://modal.com) — deploy it with:
```bash
modal deploy pricer_service2.py
```
and a Hugging Face secret named `huggingface-secret` configured in your Modal account.

## Running

```bash
python price_is_right.py
```
This launches the Gradio dashboard, which runs the agent framework on a timer, logging each agent's activity and updating the deals table as opportunities are found.

To run the agent framework directly without the UI:
```bash
python deal_agent_framework.py
```

## Tech stack

Python · Llama 3.2 (QLoRA fine-tuning via PEFT) · OpenAI / Anthropic APIs · Modal (GPU serving) · PyTorch · Transformers · Chroma · Sentence-Transformers · Gradio · Plotly · LiteLLM