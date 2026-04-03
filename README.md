<div align="center">
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyPI](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/highrize/)
  [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
</div>

# HighRize

**Universal AI token and cost compressor** — the high-performance middle-layer for your AI stack.  
`HighRize` works with any LLM API or locally hosted model to compress text, images, video, audio, and documents, slash costs, and provide deep visibility into your savings.

```bash
pip install highrize
```

---

## ⚡ Why HighRize?

In the age of long-context LLMs and high-res vision, every token is a liability.  
`HighRize` sits between your application and the AI provider, intelligently compressing payloads before they hit the wire.

- **💰 Massive Cost Savings**: Reduce your API bills by 50-90%.
- **🚀 Reduced Latency**: Smaller payloads mean faster round-trips.
- **📊 Real-time Reports**: Built-in session tracking and cost estimation.
- **🛠️ Zero Dependencies**: The core engine is lightweight and dependency-free.
- **🔌 Drop-in Ready**: Middleware and client wrappers for OpenAI, Anthropic, and more.

---

## 🚀 Quick Start

```python
from highrize import HighRize

# Initialize the engine
hr = HighRize(model="gpt-4o", provider="openai")

# Compress long text
prompt = "Please explain the concept of quantum entanglement. " * 10
result = hr.compress(prompt)

print(f"Compressed: {result.text}")
print(hr.report.summary())
# Output: 740 → 160 tokens | 78.4% saved | $0.004 saved
```

---

## 🤝 Drop-in Integration

Integrate with your favorite clients in one line. No logic changes required.

### OpenAI / Ollama / LM Studio
```python
from openai import OpenAI
from highrize import CompressedClient

client = CompressedClient(OpenAI(), model="gpt-4o")

# Same as standard OpenAI calls
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Your long prompt here..."}]
)

print(client.hr.report.summary())
```

### Anthropic
```python
from anthropic import Anthropic
from highrize import CompressedClient

client = CompressedClient(Anthropic(), provider="anthropic", model="claude-3-5-sonnet")

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}]
)
```

---

## 🏗️ Modality Engines

`HighRize` features specialized compressors for different data types.

| Modality | Features |
| :--- | :--- |
| **Text** | Filler removal, deduplication, few-shot pruning. |
| **Image** | Smart resizing, quality optimization, low-detail mode. |
| **Video** | Scene-change detection, keyframe extraction. |
| **Audio** | Silence removal, local/remote transcription. |
| **Document** | PDF/HTML/DOCX parsing with target token budgeting. |

---

## 📦 Installation

Mix and match dependencies based on your needs:

```bash
# Core only (Text)
pip install highrize

# With Image support
pip install "highrize[image]"

# With Video support
pip install "highrize[video]"

# With Audio support (Whisper + pydub)
pip install "highrize[audio]"

# Full suite
pip install "highrize[all]"
```

---

## 📜 License

HighRize is released under the **MIT License**. Build something great!
