# Financial Transaction Consultation Agent

Interview-demo implementation for the assignment: reconcile two transaction sheets, add a local RAG retriever over transaction rows, and answer user questions through one agent flow with citations.

## What It Builds

- Upload two CSV/XLS/XLSX files.
- Normalize dates, references, counterparties, descriptions, inflows, and outflows.
- Create row-level transaction IDs such as `A-2` and `B-5`.
- Match transactions by reference, date, counterparty/description similarity, and amount.
- Detect matched, missing, mismatched, duplicate, unusual, and manual-review transactions.
- Convert rows and reconciliation findings into retrievable text chunks.
- Answer questions with cited row IDs and confidence/review flags.
- Optionally use OpenAI if `OPENAI_API_KEY` is set and the `openai` package is installed.

## Tech Stack

- **Python**: core app and reconciliation engine.
- **Pandas + openpyxl**: CSV and Excel ingestion.
- **Standard library HTTP server**: demo UI without needing Streamlit/FastAPI.
- **difflib**: fuzzy matching for references and counterparties.
- **Local bag-of-words retriever**: lightweight RAG over row/finding chunks.
- **OpenAI Responses API, optional**: LLM answer synthesis using retrieved evidence only.

## Run

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

For the bundled runtime in this Codex workspace:

```powershell
& 'C:\Users\markw\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' app.py
```

## Optional LLM Setup

Install the OpenAI package if needed:

```powershell
pip install openai
```

Set credentials before running:

```powershell
$env:OPENAI_API_KEY='your_api_key'
$env:OPENAI_MODEL='gpt-4.1-mini'
python app.py
```

The app still runs without this setup. In that case, it uses the same retrieved evidence and a deterministic local answer generator.

## Demo Flow

1. Upload `sample_data/sheet_a_internal_ledger.csv` as Sheet A.
2. Upload `sample_data/sheet_b_external_statement.csv` as Sheet B.
3. Ask:
   - `Find all mismatched transactions.`
   - `Why is the balance different between both sheets?`
   - `Show transactions above 50000 only in Sheet A.`
   - `Give me a reconciliation summary.`
   - `Which transactions need manual review?`

## Interview Talking Points

- The reconciliation engine is deterministic, which makes the financial controls auditable.
- The RAG layer gives the agent grounded evidence instead of asking the LLM to inspect whole spreadsheets.
- The LLM is used for explanation, not for calculating balances or deciding matches.
- Every answer includes transaction IDs or row citations so reviewers can trace the output.
- The design can be upgraded to a production stack with FastAPI, Postgres, pgvector, background jobs, and role-based access control.
