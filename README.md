# Telco Network Troubleshooting Chatbot (LLM + RAG Prototype)

A simple, fully local chatbot that demonstrates the "Talk to Your Network" concept using a small LLM, Retrieval-Augmented Generation (RAG), and synthetic telecom logs.

You can ask natural-language questions about simulated network issues (e.g., "Why is latency high on Core-Router01?" or "Show me all ISIS neighbor drops") and get grounded answers with root-cause suggestions and recommended actions.

**This is a prototype / proof-of-concept** — it runs entirely on your laptop, uses fake data, and is not meant for production networks.

## Features

- Conversational troubleshooting interface (Streamlit)
- RAG over synthetic syslog-style logs + a small telco knowledge base
- Local LLM inference with Ollama (no cloud, no API keys)
- Fast local embeddings (Hugging Face sentence-transformers)
- Easy data regeneration for testing different scenarios

## Demo

<img src="https://github.com/user-attachments/assets/f8408b40-27d1-4850-afe6-b229f15d3e40" alt="chatbot demo" width="800"/>

## Quick Start

### 1. Install Ollama

Download and install from https://ollama.com

Pull a small model (recommended for speed on laptops):

```bash
ollama pull llama3.2:3b
```

(Other small models like phi3:mini or gemma2:2b also work well)

### 2. Set Up the Python Environment

```bash
git clone https://github.com/yourusername/telco-troubleshooting-chatbot.git
cd telco-troubleshooting-chatbot
pip install -r requirements.txt
```

### 3. Generate Synthetic Logs

```bash
python data_generator.py
```

This creates `data/simulated_logs.csv` with realistic telecom-style alerts, KPIs, and actions.

### 4. Run the Chatbot

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501 and start asking questions.

## Example Questions to Try

- "Show me all critical alerts"
- "What caused the ISIS neighbor drops?"
- "Any logs with high packet loss?"
- "Recommend actions for interface flaps"
- "Why is latency high?"

## Project Structure

```
.
├── app.py                  # Main Streamlit app + RAG chain
├── data_generator.py       # Generates synthetic logs
├── data/
│   ├── simulated_logs.csv  # Generated logs
│   └── telco_manual.txt    # Domain knowledge for RAG
├── requirements.txt
└── README.md
```

## Limitations

- Fully synthetic data — no real network integration
- Small local models can be less accurate than giant cloud LLMs
- No persistence beyond the current session
- Prototype only — not hardened for production use

## Future Ideas (Contributions Welcome)

- Add agentic workflows (e.g., auto-diagnostics)
- Connect to network simulators (Mininet, GNS3)
- Support real (anonymized) log ingestion
- Fine-tune a telco-specific small model

## License

MIT License — feel free to fork, modify, and use.

---

Built as a personal project to explore LLM applications in telecom network operations.
