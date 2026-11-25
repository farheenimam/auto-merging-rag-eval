# auto-merging-rag-eval

A RAG system over PDF documents that uses auto-merging retrieval and cross-encoder reranking. It compares a 2-layer and a 3-layer chunk hierarchy using the RAG triad, and includes a Streamlit app to see both side by side.

I tested it on a software requirements document (SRS) for a project called Verixa.

## Demo

Watch the demo: [demo.mp4](demo.mp4)

## Techniques used

**1. Auto-merging retrieval (hierarchical chunking)**
- The document is split into layers of parent and child chunks.
  - 2-layer: 2048 and 512 tokens
  - 3-layer: 2048, 512 and 128 tokens
- Only the smallest (leaf) chunks are embedded and indexed. The parent chunks are kept in a docstore.
- At query time the top 12 leaf chunks are fetched by vector similarity. If enough of them share the same parent, they are replaced by that parent chunk, so the LLM gets more complete context.

**2. Cross-encoder reranking**
- A cross-encoder takes the question together with each retrieved chunk and gives it a relevance score.
- The top 6 chunks are kept.

**3. Synthesis**
- The LLM writes the answer from those 6 chunks.

Pipeline: ingestion (load, chunk, embed, index) -> retrieval (similarity search, auto-merge, rerank) -> synthesis (LLM answer).

## How it is evaluated

The RAG triad, scored by an LLM judge. Each score goes from 0 to 10 and is rescaled to 0-1.

| Metric | What it checks |
|---|---|
| Context relevance | Are the retrieved chunks relevant to the question? Each chunk is scored, then averaged. |
| Groundedness | Is the answer supported by the retrieved chunks, or made up? |
| Answer relevance | Does the answer actually address the question? |

The judge is a different model from the generator, to avoid a model grading its own output.

Both configurations (2 layers and 3 layers) run on the same questions, and the averages are compared.

## Speed

Response time is recorded for every query (the `seconds` column in `results.csv`), so the 2-layer and 3-layer setups can be compared on latency as well as quality. The reranker and embedding model run locally, and generation runs on Groq.

Results from my run (fill in after running `eval_triad.py`):

| Layers | Context relevance | Groundedness | Answer relevance | Avg seconds |
|---|---|---|---|---|
| 2 layers | | | | |
| 3 layers | | | | |

## Built with

- LlamaIndex 0.9.48 (hierarchical node parser, auto-merging retriever)
- Groq API: `openai/gpt-oss-120b` as generator, `openai/gpt-oss-20b` as judge
- `BAAI/bge-small-en-v1.5` for embeddings
- `BAAI/bge-reranker-base` for reranking
- Streamlit for the frontend
- pypdf, pandas
- Google Colab

## Files

- `rag_core.py`: indexing, retrieval, query engine and the triad scorer
- `eval_triad.py`: runs the questions on both configs and saves `results.csv` and `summary.csv`
- `app.py`: Streamlit app that shows both answers, the retrieved chunks and the triad scores
- `demo.mp4`: screen recording of the app

## Setup

```bash
pip install llama-index==0.9.48 sentence-transformers pypdf streamlit pandas
export GROQ_API_KEY=your_key
export JUDGE_MODEL=openai/gpt-oss-20b
```

Run the evaluation (edit `QUESTIONS` in `eval_triad.py` first):

```bash
python eval_triad.py path/to/your.pdf
```

Run the app:

```bash
streamlit run app.py
```

Upload a PDF in the sidebar and ask questions. Both configs answer each question, and the app shows the retrieved chunks and triad scores for each.

## Notes

- Indexes are saved to `idx_*` folders. Delete them if you change the document or chunk sizes.
- The judge is an LLM, so scores are approximate. Use 10 or more questions before drawing conclusions.
- Groq changes its model list often. If a model returns a 404, pick another from `https://api.groq.com/openai/v1/models`.
