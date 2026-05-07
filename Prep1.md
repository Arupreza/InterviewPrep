# Boeing AI Application Engineer — 20 Coding Questions & Solutions

**Interview:** May 11, 2026 | 60 min Technical + 30 min Behavioral
**Goal:** Master these 20 questions cold. They cover every coding angle for this JD.

---

## SECTION A — PYTHON & DATA PROCESSING (Q1–Q3)

### Q1. Read CSV, clean, and compute basic stats

```python
import pandas as pd

df = pd.read_csv("sensor_data.csv")
df = df.dropna()                              # remove missing rows
df = df[df["temperature"] > 0]                # filter invalid
summary = df.groupby("aircraft_id")["temperature"].agg(["mean", "std", "max"])
print(summary)
```

**Why asked:** Manufacturing/factory data is messy. Boeing wants to see you handle real data.

---

### Q2. Normalize numerical features (min-max scaling)

```python
import numpy as np

def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0) + 1e-8)

data = np.array([[1, 100], [2, 200], [3, 300]], dtype=np.float32)
print(min_max_normalize(data))
```

**Why asked:** Preprocessing is daily ML work. The `+1e-8` shows production awareness.

---

### Q3. Compute classification metrics from scratch

```python
def classification_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy  = (tp + tn) / len(y_true)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

print(classification_metrics([1,0,1,1,0], [1,1,1,0,0]))
```

**Why asked:** Tests if you understand what your models actually output.

---

## SECTION B — PYTORCH MODEL DEVELOPMENT (Q4–Q6)

### Q4. Build and train a simple classifier

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, in_dim=10, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# One training step
x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()
print(f"Loss: {loss.item():.4f}")
```

**Why asked:** Boeing wants to see if you actually know PyTorch — not just import it.

---

### Q5. Save and load a model checkpoint

```python
import torch

# Save
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 10,
}, "checkpoint.pt")

# Load
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()
```

**Why asked:** Production ML = checkpointing. Tests if you've shipped models, not just trained them.

---

### Q6. Run inference with proper eval mode

```python
import torch

def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(input_data, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)        # add batch dimension
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
    return pred.tolist(), probs.tolist()

print(predict(model, [0.1] * 10))
```

**Why asked:** `model.eval()` + `torch.no_grad()` = production-ready inference. Forgetting these is a red flag.

---

## SECTION C — ONNX & MODEL SERVING (Q7–Q9)

### Q7. Export PyTorch model to ONNX

```python
import torch

model.eval()
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17
)
print("ONNX export done")
```

**Why asked:** ONNX is explicit in JD. `dynamic_axes` for batching shows production knowledge.

---

### Q8. ONNX Runtime inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

x = np.random.randn(1, 10).astype(np.float32)
output = session.run([output_name], {input_name: x})[0]
print(f"Prediction: {np.argmax(output)}")
```

**Why asked:** This is the JD-named runtime. Critical to know cold.

---

### Q9. Compare PyTorch vs ONNX latency

```python
import time
import numpy as np
import torch
import onnxruntime as ort

x_np = np.random.randn(1, 10).astype(np.float32)
x_torch = torch.from_numpy(x_np)
session = ort.InferenceSession("model.onnx")

# PyTorch timing
model.eval()
start = time.perf_counter()
with torch.no_grad():
    for _ in range(100):
        _ = model(x_torch)
torch_ms = (time.perf_counter() - start) * 10  # ms per run

# ONNX timing
start = time.perf_counter()
for _ in range(100):
    _ = session.run(None, {"input": x_np})
onnx_ms = (time.perf_counter() - start) * 10

print(f"PyTorch: {torch_ms:.2f} ms | ONNX: {onnx_ms:.2f} ms")
```

**Why asked:** Performance optimization is a preferred qual. Shows you measure, don't guess.

---

## SECTION D — FASTAPI BACKEND (Q10–Q12)

### Q10. Production FastAPI inference endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
import numpy as np

app = FastAPI(title="Aircraft Defect Classifier")
session = ort.InferenceSession("model.onnx")

class InferenceRequest(BaseModel):
    features: list[float] = Field(..., min_length=10, max_length=10)

class InferenceResponse(BaseModel):
    prediction: int
    confidence: float

@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    try:
        arr = np.array([req.features], dtype=np.float32)
        output = session.run(None, {"input": arr})[0]
        return InferenceResponse(
            prediction=int(np.argmax(output)),
            confidence=float(np.max(output))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Why asked:** This is THE core question for this role. Pydantic validation + health check = production-ready.

---

### Q11. Async endpoint with background task

```python
from fastapi import FastAPI, BackgroundTasks
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)

def log_prediction(features, prediction):
    logging.info(f"Logged: {features} -> {prediction}")

@app.post("/predict_async")
async def predict_async(features: list[float], bg: BackgroundTasks):
    prediction = sum(features) > 0   # toy logic
    bg.add_task(log_prediction, features, prediction)
    return {"prediction": prediction}
```

**Why asked:** Tests async understanding + how to log without blocking the response.

---

### Q12. File upload endpoint (e.g., document for RAG)

```python
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    word_count = len(text.split())
    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "word_count": word_count
    }
```

**Why asked:** Aircraft manuals = lots of documents. Upload + processing is realistic.

---

## SECTION E — LLM & AGENTS (Q13–Q15)

### Q13. Basic LLM API call

```python
import anthropic

client = anthropic.Anthropic()

def ask_llm(question: str, system: str = None) -> str:
    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": question}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return response.content[0].text

print(ask_llm("Explain ONNX in 2 sentences."))
```

**Why asked:** Baseline LLM integration. You should write this in 90 seconds.

---

### Q14. LLM Agent with tool calling

```python
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_part_inventory",
    "description": "Get current inventory count for an aircraft part",
    "input_schema": {
        "type": "object",
        "properties": {"part_id": {"type": "string"}},
        "required": ["part_id"]
    }
}]

# Mock tool implementation
def get_part_inventory(part_id: str) -> dict:
    return {"part_id": part_id, "count": 42, "location": "Warehouse A"}

def run_agent(query: str):
    messages = [{"role": "user", "content": query}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

        if response.stop_reason == "tool_use":
            tool_use = next(b for b in response.content if b.type == "tool_use")
            result = get_part_inventory(**tool_use.input)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(result)
                }]
            })

print(run_agent("How many of part B737-WING-01 do we have?"))
```

**Why asked:** This is your differentiator. Stratum-MoE experience maps directly. Walk through it confidently.

---

### Q15. Streaming LLM response

```python
import anthropic

client = anthropic.Anthropic()

def stream_response(query: str):
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": query}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()

stream_response("List 5 aircraft safety checks.")
```

**Why asked:** Production LLM apps stream. Shows you've shipped real LLM products.

---

## SECTION F — RAG & EMBEDDINGS (Q16–Q17)

### Q16. Chunk text + retrieve top-k

```python
import numpy as np

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def retrieve_top_k(query_emb, doc_embs, docs, k=3):
    q = query_emb / np.linalg.norm(query_emb)
    d = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    scores = d @ q
    top_idx = np.argsort(scores)[-k:][::-1]
    return [(docs[i], float(scores[i])) for i in top_idx]

# Demo
docs = ["Boeing 737 MAX manual", "Engine inspection guide", "Wing assembly spec"]
doc_embs = np.random.randn(3, 384)
query_emb = np.random.randn(384)
print(retrieve_top_k(query_emb, doc_embs, docs))
```

**Why asked:** Aviation = document-heavy. RAG over manuals is a likely Boeing use case.

---

### Q17. End-to-end RAG with LLM

```python
import anthropic
import numpy as np

client = anthropic.Anthropic()

def rag_answer(query: str, retrieved_docs: list[str]) -> str:
    context = "\n\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(retrieved_docs))

    prompt = f"""Use only the following context to answer.
If not found, say "Not in context."

Context:
{context}

Question: {query}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

retrieved = [
    "Boeing 737 has two CFM56 engines.",
    "Maximum takeoff weight is 79,000 kg."
]
print(rag_answer("What engines does the 737 use?", retrieved))
```

**Why asked:** Combines retrieval + generation. End-to-end RAG = end-to-end AI solution (JD requirement).

---

## SECTION G — PERFORMANCE & ASYNC (Q18–Q19)

### Q18. Async batch inference

```python
import asyncio
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")

async def predict_one(features):
    loop = asyncio.get_event_loop()
    arr = np.array([features], dtype=np.float32)
    output = await loop.run_in_executor(
        None, lambda: session.run(None, {"input": arr})
    )
    return int(np.argmax(output[0]))

async def batch_predict(batch):
    return await asyncio.gather(*[predict_one(f) for f in batch])

batch = [[0.1]*10, [0.2]*10, [0.3]*10]
print(asyncio.run(batch_predict(batch)))
```

**Why asked:** Performance optimization is a preferred qual. Async = throughput.

---

### Q19. HuggingFace Quantization (LLM-grade)

This is the modern stack for quantizing LLMs and HuggingFace models. Master all five methods.

---

**19a. 4-bit Quantization (QLoRA workflow — most important)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                  # NormalFloat4 — optimal for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,      # compute in bf16
    bnb_4bit_use_double_quant=True,             # quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
```

**Memory:** ~5 GB vs ~16 GB FP16. Used in QLoRA fine-tuning.

---

**19b. 8-bit Quantization (when 4-bit hurts accuracy)**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,                     # outlier threshold
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Use when:** 4-bit degrades task accuracy noticeably. ~8 GB memory.

---

**19c. GPTQ (Post-Training Quantization — fastest inference)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",                               # calibration dataset
    tokenizer=tokenizer,
    group_size=128,
    desc_act=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=gptq_config,
    device_map="auto",
)
model.save_pretrained("./llama-gptq-4bit")
```

**Use when:** Production inference, calibration data available. Faster than bitsandbytes.

---

**19d. AWQ (Activation-aware — best accuracy at 4-bit)**

```python
from transformers import AutoModelForCausalLM, AwqConfig

# Load pre-quantized AWQ model
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ",
    device_map="auto",
)
```

**Use when:** Accuracy-critical inference. Preserves salient weights better than GPTQ.

---

**19e. ONNX Quantization via Optimum (edge/CPU serving)**

```python
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Convert HF model to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    export=True,
)
model.save_pretrained("./onnx-model")

# Quantize to INT8
quantizer = ORTQuantizer.from_pretrained("./onnx-model")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

quantizer.quantize(
    save_dir="./onnx-model-quantized",
    quantization_config=qconfig,
)
```

**Use when:** CPU-only deployment, edge inference, factory hardware.

---

**Decision Matrix**

| Method | Bits | Use Case | Speed | Accuracy |
|---|---|---|---|---|
| **bitsandbytes NF4** | 4 | Fine-tuning (QLoRA), quick load | Medium | Good |
| **bitsandbytes 8-bit** | 8 | When 4-bit hurts task | Slow | Better |
| **GPTQ** | 4 | Production inference | Fast | Good |
| **AWQ** | 4 | Accuracy-critical inference | Fast | Best |
| **ONNX INT8 (Optimum)** | 8 | Edge / CPU serving | Fastest on CPU | Good |

---

**Verbal answer to memorize for the interview:**

> *"NF4 is information-theoretically optimal for normally-distributed weights, which is why QLoRA uses it. Double quantization quantizes the quantization constants themselves, saving roughly 0.4 bits per parameter. Compute stays in bf16 to preserve gradient quality during the backward pass. For production inference I'd use GPTQ or AWQ — AWQ when accuracy matters more, GPTQ when speed does. For edge or CPU deployment, ONNX INT8 via Optimum is the right choice."*

**Why asked:** Quantization is the bridge between your QLoRA fine-tuning experience and Boeing's edge/factory deployment needs. This single answer covers fine-tuning, server inference, and edge — the full spectrum.

---

## SECTION H — DEPLOYMENT (Q20)

### Q20. Dockerfile + docker-compose for FastAPI ML service

**Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**
```yaml
version: "3.8"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.onnx
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: ml_db
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

**Why asked:** Docker-Compose is JD basic qual. Memorize this structure.

---

## SECTION I — RAG EVALUATION METRICS (Q21)

### Q21. RAG Evaluation — Faithfulness, Relevancy, Context Recall

---

#### Concept First (What Boeing May Ask Verbally)

| Metric | What It Measures | Formula / Logic |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | claims in answer supported by context / total claims |
| **Answer Relevancy** | Does the answer address the question? | cosine similarity(question, answer embedding) |
| **Context Precision** | Are retrieved chunks actually useful? | relevant chunks in top-k / total retrieved chunks |
| **Context Recall** | Did retrieval capture all needed info? | relevant chunks retrieved / total relevant chunks |

---

#### Code: Faithfulness Score (Manual)

```python
def faithfulness_score(answer: str, context: str) -> float:
    """
    Simplified: check what fraction of answer sentences
    are supported by the context.
    In production use RAGAS or DeepEval.
    """
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    supported = sum(1 for s in sentences if s.lower() in context.lower())
    return supported / len(sentences) if sentences else 0.0

context = "The Boeing 737 uses CFM56 engines. Max range is 6,500 km."
answer = "The 737 uses CFM56 engines. It can carry 200 passengers."
print(faithfulness_score(answer, context))  # 0.5 — one claim unsupported
```

---

#### Code: Answer Relevancy (Embedding Cosine)

```python
import numpy as np

def answer_relevancy(question_emb: np.ndarray, answer_emb: np.ndarray) -> float:
    q = question_emb / np.linalg.norm(question_emb)
    a = answer_emb / np.linalg.norm(answer_emb)
    return float(np.dot(q, a))

# Simulated embeddings
q_emb = np.random.randn(384)
a_emb = np.random.randn(384)
print(f"Answer Relevancy: {answer_relevancy(q_emb, a_emb):.4f}")
```

---

#### Code: Context Precision & Recall

```python
def context_precision(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    hits = sum(1 for r in retrieved if r in relevant_set)
    return hits / len(retrieved) if retrieved else 0.0

def context_recall(retrieved: list[str], relevant: list[str]) -> float:
    retrieved_set = set(retrieved)
    hits = sum(1 for r in relevant if r in retrieved_set)
    return hits / len(relevant) if relevant else 0.0

retrieved = ["chunk_1", "chunk_2", "chunk_3"]
relevant  = ["chunk_1", "chunk_3", "chunk_5"]

print(f"Precision: {context_precision(retrieved, relevant):.2f}")  # 2/3 = 0.67
print(f"Recall:    {context_recall(retrieved, relevant):.2f}")     # 2/3 = 0.67
```

---

#### Code: Full RAG Eval with RAGAS (Production Tool)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Prepare evaluation dataset
data = {
    "question":  ["What engines does the 737 use?"],
    "answer":    ["The 737 uses CFM56 engines."],
    "contexts":  [["The Boeing 737 uses CFM56 engines. Max range is 6,500 km."]],
    "ground_truth": ["CFM56 engines power the Boeing 737."]
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(results)
```

**Why asked:** If Boeing asks about evaluating RAG pipelines (aviation manuals, compliance docs), this is the answer. RAGAS is the industry standard evaluation framework.

---

## SECTION J — AGENT EVALUATION METRICS (Q22)

### Q22. Agent Evaluation — Task Success, Trajectory Quality, Tool Use

---

#### Concept First (What Boeing May Ask Verbally)

| Metric | What It Measures | How |
|---|---|---|
| **Task Success Rate** | Did agent complete the goal? | Binary pass/fail on final output |
| **Tool Call Accuracy** | Did agent call the right tools? | Correct tool & args / total steps |
| **Trajectory Efficiency** | Did agent take the shortest path? | Optimal steps / actual steps |
| **Hallucination Rate** | Did agent invent tool results? | Claims not grounded in tool output |
| **Latency per Task** | End-to-end time to complete task | Clock time per agent run |

---

#### Code: Task Success Rate

```python
def task_success_rate(results: list[dict]) -> float:
    """
    results: list of {"expected": ..., "actual": ..., "success": bool}
    """
    successes = sum(1 for r in results if r["success"])
    return successes / len(results) if results else 0.0

results = [
    {"task": "get part count", "success": True},
    {"task": "file maintenance report", "success": True},
    {"task": "schedule inspection", "success": False},
]
print(f"Task Success Rate: {task_success_rate(results):.2f}")  # 0.67
```

---

#### Code: Tool Call Accuracy

```python
def tool_call_accuracy(
    expected_calls: list[dict],
    actual_calls: list[dict]
) -> float:
    """
    Compare expected vs actual tool calls (name + key args).
    """
    correct = 0
    for exp, act in zip(expected_calls, actual_calls):
        if exp["name"] == act["name"] and exp["args"] == act["args"]:
            correct += 1
    return correct / len(expected_calls) if expected_calls else 0.0

expected = [{"name": "get_part_inventory", "args": {"part_id": "B737-01"}}]
actual   = [{"name": "get_part_inventory", "args": {"part_id": "B737-01"}}]
print(f"Tool Accuracy: {tool_call_accuracy(expected, actual):.2f}")  # 1.0
```

---

#### Code: Trajectory Efficiency

```python
def trajectory_efficiency(optimal_steps: int, actual_steps: int) -> float:
    """
    1.0 = perfect. <1.0 = agent took unnecessary steps.
    """
    return optimal_steps / actual_steps if actual_steps > 0 else 0.0

print(trajectory_efficiency(optimal_steps=3, actual_steps=5))  # 0.6
```

---

#### Code: End-to-End Agent Evaluation Runner

```python
import time
import anthropic

client = anthropic.Anthropic()

def evaluate_agent(test_cases: list[dict], agent_fn) -> dict:
    """
    test_cases: [{"query": str, "expected_tool": str, "validate_fn": callable}]
    agent_fn: your agent function
    """
    results = {
        "total": len(test_cases),
        "success": 0,
        "total_latency_ms": 0,
        "tool_hits": 0,
    }

    for case in test_cases:
        start = time.perf_counter()
        response = agent_fn(case["query"])
        latency = (time.perf_counter() - start) * 1000

        results["total_latency_ms"] += latency

        if case["validate_fn"](response):
            results["success"] += 1

    results["success_rate"] = results["success"] / results["total"]
    results["avg_latency_ms"] = results["total_latency_ms"] / results["total"]
    return results


# Example usage
def mock_agent(query):
    return {"tool_called": "get_part_inventory", "result": {"count": 42}}

test_cases = [
    {
        "query": "How many of part B737-01 do we have?",
        "expected_tool": "get_part_inventory",
        "validate_fn": lambda r: r["tool_called"] == "get_part_inventory"
    }
]

print(evaluate_agent(test_cases, mock_agent))
```

---

#### Production Tool: Use DeepEval for Agent Evaluation

```python
from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="How many B737-01 parts do we have?",
    actual_output="There are 42 units of B737-01 in Warehouse A.",
    expected_tools=[ToolCall(name="get_part_inventory")],
    tools_called=[ToolCall(name="get_part_inventory", output={"count": 42})]
)

evaluate(
    test_cases=[test_case],
    metrics=[TaskCompletionMetric(), ToolCorrectnessMetric()]
)
```

**Why asked:** If Boeing builds production agents for factory automation, they need to evaluate them. Knowing RAGAS + DeepEval signals you've shipped, not just built demos.

---

## TOP 6 TO MEMORIZE COLD

| Rank | Question | Why |
|---|---|---|
| 🔴 1 | Q10 — FastAPI + ONNX endpoint | Single most likely question |
| 🔴 2 | Q8 — ONNX Runtime inference | JD explicitly names ONNX |
| 🔴 3 | Q14 — LLM Agent with tool calling | Your differentiator |
| 🔴 4 | Q7 — ONNX export | Pairs with Q8 |
| 🔴 5 | Q20 — Docker-Compose | JD basic qual |
| 🔴 6 | Q17 — End-to-end RAG | Aviation use case fit |

---

## HOW TO USE THIS DOCUMENT

1. **Day 1–2:** Type every solution from memory. Find your weak spots.
2. **Day 3:** Drill the top 6 until you can write them in <2 minutes each.
3. **Day 4 (May 10):** Light review only. Do NOT cram. Sleep early.
4. **May 11:** Walk in confident. You match this JD better than most candidates.

---

**You are not bluffing. You have done all of this in production.**
**The interview is about communicating what you already know.**