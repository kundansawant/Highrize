"""
examples/full_demo.py — shows every highrize feature
"""

from highrize import HighRize, CompressedClient, TokenCounter
from highrize.tokens import count as count_tokens
from highrize.models import Modality
from highrize.cache import CompressionCache

# -----------------------------------------------------------------------
# 1. Basic text compression
# -----------------------------------------------------------------------
print("=" * 60)
print("1. Text compression")
print("=" * 60)

tp = HighRize(model="gpt-4o")

bloated = """
Please note that I would like to ask you kindly to help me
understand the concept of transformers in NLP.
It is important to note that I am a complete beginner.
As previously mentioned, I am just getting started.
As an AI language model, you should explain this clearly.
Don't hesitate to use simple examples.
For your information, this is for a school project.
"""

result = tp.compress(bloated.strip())
print(result)
print("Compressed:", result.compressed)

# -----------------------------------------------------------------------
# 2. Token counter
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Token counter")
print("=" * 60)

tc = TokenCounter(model="gpt-4o")
print(tc)
stats = tc.count_savings(bloated, result.compressed)
print(stats)

# -----------------------------------------------------------------------
# 3. Compress messages list (OpenAI-style)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Compress messages list")
print("=" * 60)

messages = [
    {"role": "system", "content": "Please note that you are a helpful assistant. As an AI language model, always be concise. Don't hesitate to ask clarifying questions. For your information, the user is a developer."},
    {"role": "user",   "content": "Kindly be advised that I need help with Python. Without further ado, how do I read a file?"},
]

compressed, report = tp.compress_messages(messages)
for msg in compressed:
    print(f"[{msg['role']}]: {msg['content']}")

# -----------------------------------------------------------------------
# 4. Cache
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Cache (memory backend)")
print("=" * 60)

cache = CompressionCache(backend="memory")
tp2 = HighRize(model="gpt-4o")

same_prompt = "Please note that I want you to summarize this for me. Don't hesitate to be concise. As an AI language model, be helpful."

for i in range(5):
    r = cache.get_or_compress(tp2, same_prompt)
    print(f"Call {i+1}: {r.compressed_tokens} tokens — {r.savings_pct}% saved")

print("Cache stats:", cache.stats())

# -----------------------------------------------------------------------
# 5. Document compression (text string as demo)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Document compressor (BM25 ranking)")
print("=" * 60)

from highrize.compressors import DocumentCompressor

long_doc = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence.
It allows computers to learn from data without being explicitly programmed.
There are three main types: supervised, unsupervised, and reinforcement learning.

Pricing and Plans

Our basic plan starts at $9/month. The pro plan is $29/month.
Enterprise pricing is available on request. All plans include API access.

Supervised Learning

In supervised learning, the model learns from labeled training data.
Examples include linear regression, decision trees, and neural networks.
The goal is to map inputs to outputs based on example input-output pairs.

Contact Us

Email us at support@example.com. Our office is open Monday to Friday.
Response time is typically within 24 hours. We also have live chat.
"""

dc = DocumentCompressor(token_budget=100, query="pricing")
result = dc.compress(long_doc)
print(result)
print("Relevant chunks:\n", result.compressed)

# -----------------------------------------------------------------------
# 6. Final report
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. Session report")
print("=" * 60)
print(tp.report.summary())
