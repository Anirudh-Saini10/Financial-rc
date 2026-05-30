import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from datetime import datetime
from rapidfuzz import fuzz
from sklearn.ensemble import IsolationForest

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configurations & Defaults ---
st.set_page_config(page_title="Reconciliation Agent", layout="wide")

COLUMN_ALIASES = {
    "date": {"date", "transaction date", "txn date", "value date", "posting date"},
    "reference": {"reference", "ref", "transaction id", "txn id", "utr", "cheque no", "invoice", "id"},
    "description": {"description", "narration", "details", "memo", "particulars", "remarks"},
    "counterparty": {"counterparty", "party", "vendor", "customer", "beneficiary", "payee", "payer", "name"},
    "amount": {"amount", "transaction amount", "value", "net amount"},
    "debit": {"debit", "withdrawal", "paid out", "outflow"},
    "credit": {"credit", "deposit", "paid in", "inflow"},
    "type": {"type", "transaction type", "dr/cr", "direction"},
}

# --- Data Normalization ---
def normalize_label(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

def compact_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value or "")).strip()

def amount_to_float(value) -> float:
    if pd.isna(value):
        return 0.0
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in {"", ".", "-", "-."}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def parse_date(value) -> pd.Timestamp:
    text = compact_text(value)
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", text):
        return pd.to_datetime(text, errors="coerce")
    return pd.to_datetime(text, dayfirst=True, errors="coerce")

def detect_columns(df: pd.DataFrame) -> dict:
    normalized = {normalize_label(col): col for col in df.columns}
    detected = {key: None for key in COLUMN_ALIASES}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                detected[canonical] = normalized[alias]
                break
    return detected

def signed_amount(row: pd.Series, cols: dict) -> float:
    debit_col = cols.get("debit")
    credit_col = cols.get("credit")
    if debit_col or credit_col:
        debit = amount_to_float(row.get(debit_col)) if debit_col else 0.0
        credit = amount_to_float(row.get(credit_col)) if credit_col else 0.0
        if credit and not debit:
            return credit
        if debit and not credit:
            return -abs(debit)
        if credit or debit:
            return credit - debit

    amount = amount_to_float(row.get(cols.get("amount"))) if cols.get("amount") else 0.0
    tx_type = normalize_label(row.get(cols.get("type"), "")) if cols.get("type") else ""
    if tx_type in {"debit", "dr", "withdrawal", "outflow", "paid out"}:
        return -abs(amount)
    if tx_type in {"credit", "cr", "deposit", "inflow", "paid in"}:
        return abs(amount)
    return amount

def normalize_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    cols = detect_columns(df)
    records = []
    for index, row in df.iterrows():
        date_raw = row.get(cols.get("date")) if cols.get("date") else None
        parsed_date = parse_date(date_raw)
        amount = signed_amount(row, cols)
        description = compact_text(row.get(cols.get("description"))) if cols.get("description") else ""
        counterparty = compact_text(row.get(cols.get("counterparty"))) if cols.get("counterparty") else ""
        reference = compact_text(row.get(cols.get("reference"))) if cols.get("reference") else ""
        combined_party = counterparty or description

        normalized_ref = normalize_label(reference)
        normalized_party = normalize_label(combined_party)
        date_key = parsed_date.strftime("%Y-%m-%d") if not pd.isna(parsed_date) else ""
        direction = "inflow" if amount >= 0 else "outflow"
        row_no = int(index) + 2

        records.append({
            "sheet": sheet_name,
            "row_number": row_no,
            "transaction_id": f"{sheet_name}-{row_no}",
            "date": date_key,
            "reference": reference,
            "description": description,
            "counterparty": counterparty,
            "amount": round(float(amount), 2),
            "abs_amount": round(abs(float(amount)), 2),
            "direction": direction,
            "normalized_ref": normalized_ref,
            "normalized_party": normalized_party,
        })
    return pd.DataFrame(records)

def load_sheet(uploaded_file, sheet_name: str) -> pd.DataFrame:
    if uploaded_file.name.endswith('.csv'):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
    
    if raw.empty:
        raise ValueError(f"{sheet_name} contains no data.")
    return normalize_sheet(raw, sheet_name)

# --- Matching Logic ---
def date_distance(left: str, right: str) -> int:
    if not left or not right:
        return 999
    try:
        return abs((datetime.fromisoformat(left) - datetime.fromisoformat(right)).days)
    except ValueError:
        return 999

def match_score(a: pd.Series, b: pd.Series) -> tuple:
    score = 0.0
    reasons = []
    ref_a = a["normalized_ref"]
    ref_b = b["normalized_ref"]
    if ref_a and ref_b and ref_a == ref_b:
        score += 0.45
        reasons.append("same reference")
    elif ref_a and ref_b and fuzz.ratio(ref_a, ref_b) >= 86.0:
        score += 0.32
        reasons.append("similar reference")

    days = date_distance(a["date"], b["date"])
    if days == 0:
        score += 0.2
        reasons.append("same date")
    elif days <= 2:
        score += 0.12
        reasons.append("nearby date")

    party_similarity = fuzz.ratio(a["normalized_party"], b["normalized_party"])
    if party_similarity >= 90.0:
        score += 0.25
        reasons.append("same counterparty/description")
    elif party_similarity >= 72.0:
        score += 0.16
        reasons.append("similar counterparty/description")

    amount_gap = abs(float(a["abs_amount"]) - float(b["abs_amount"]))
    amount_base = max(float(a["abs_amount"]), float(b["abs_amount"]), 1.0)
    if amount_gap <= 0.01:
        score += 0.1
        reasons.append("same amount")
    elif amount_gap / amount_base <= 0.05:
        score += 0.06
        reasons.append("near amount")

    return score, reasons

# --- Anomaly Detection ---
def suspicious_transactions(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame, matched: list, mismatched: list) -> list:
    combined = pd.concat([sheet_a, sheet_b], ignore_index=True)
    suspicious = []
    if combined.empty:
        return suspicious

    # Scikit-learn Isolation Forest for Anomaly Detection
    X = combined[['abs_amount']].fillna(0).values
    # Ensure there's enough data for IsolationForest
    if len(X) > 10:
        clf = IsolationForest(contamination=0.05, random_state=42)
        combined['anomaly'] = clf.fit_predict(X)
    else:
        combined['anomaly'] = 1 # Not enough data, mark all as normal

    duplicate_keys = combined.groupby(["sheet", "date", "normalized_ref", "abs_amount"]).size()
    duplicate_keys = {key for key, count in duplicate_keys.items() if count > 1 and key[2]}

    for _, row in combined.iterrows():
        flags = []
        key = (row["sheet"], row["date"], row["normalized_ref"], row["abs_amount"])
        if key in duplicate_keys:
            flags.append("possible duplicate")
        if row.get('anomaly', 1) == -1:
            flags.append("anomalous amount detected by model")
        if not row["reference"]:
            flags.append("missing reference")
        if flags:
            suspicious.append(row.to_dict() | {"flags": ", ".join(flags), "needs_review": True})

    for item in mismatched:
        suspicious.append({
            "transaction_id": f"{item['sheet_a_id']} / {item['sheet_b_id']}",
            "sheet": "Both",
            "row_number": f"{item['sheet_a_row']} / {item['sheet_b_row']}",
            "date": item["date_a"],
            "reference": item["reference_a"] or item["reference_b"],
            "counterparty": item["party_a"],
            "amount": item["amount_a"],
            "flags": f"amount variance {item['variance']}",
            "needs_review": True,
        })
    return suspicious

# --- Reconciliation Engine ---
def reconcile(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame) -> dict:
    matched = []
    mismatched = []
    used_b = set()

    for idx_a, row_a in sheet_a.iterrows():
        best = None
        for idx_b, row_b in sheet_b.iterrows():
            if idx_b in used_b:
                continue
            score, reasons = match_score(row_a, row_b)
            if best is None or score > best[1]:
                best = (idx_b, score, reasons)
        if best and best[1] >= 0.58:
            idx_b, score, reasons = best
            used_b.add(idx_b)
            row_b = sheet_b.loc[idx_b]
            amount_variance = round(float(row_a["amount"]) - float(row_b["amount"]), 2)
            item = {
                "sheet_a_id": row_a["transaction_id"],
                "sheet_b_id": row_b["transaction_id"],
                "sheet_a_row": int(row_a["row_number"]),
                "sheet_b_row": int(row_b["row_number"]),
                "date_a": row_a["date"],
                "date_b": row_b["date"],
                "reference_a": row_a["reference"],
                "reference_b": row_b["reference"],
                "party_a": row_a["counterparty"] or row_a["description"],
                "party_b": row_b["counterparty"] or row_b["description"],
                "amount_a": float(row_a["amount"]),
                "amount_b": float(row_b["amount"]),
                "variance": amount_variance,
                "confidence": round(min(score, 0.99), 2),
                "match_reasons": ", ".join(reasons),
                "needs_review": score < 0.78 or abs(amount_variance) > 0.01,
            }
            if abs(amount_variance) <= 0.01:
                matched.append(item)
            else:
                mismatched.append(item)

    matched_a_ids = {item["sheet_a_id"] for item in matched + mismatched}
    missing_in_b = sheet_a[~sheet_a["transaction_id"].isin(matched_a_ids)].to_dict("records")
    missing_in_a = sheet_b[~sheet_b.index.isin(used_b)].to_dict("records")

    suspicious = suspicious_transactions(sheet_a, sheet_b, matched, mismatched)
    
    total_a = {"inflow": round(float(sheet_a[sheet_a["amount"] >= 0]["amount"].sum()), 2), 
               "outflow": round(float(sheet_a[sheet_a["amount"] < 0]["amount"].sum()), 2)}
    total_a["net"] = round(total_a["inflow"] + total_a["outflow"], 2)
    
    total_b = {"inflow": round(float(sheet_b[sheet_b["amount"] >= 0]["amount"].sum()), 2), 
               "outflow": round(float(sheet_b[sheet_b["amount"] < 0]["amount"].sum()), 2)}
    total_b["net"] = round(total_b["inflow"] + total_b["outflow"], 2)

    summary = {
        "sheet_a": total_a,
        "sheet_b": total_b,
        "net_variance": round(total_a["net"] - total_b["net"], 2),
        "matched_count": len(matched),
        "mismatched_count": len(mismatched),
        "missing_in_a_count": len(missing_in_a),
        "missing_in_b_count": len(missing_in_b),
        "manual_review_count": len(mismatched) + len(missing_in_a) + len(missing_in_b),
    }

    return {
        "matched": matched,
        "mismatched": mismatched,
        "missing_in_b": missing_in_b,
        "missing_in_a": missing_in_a,
        "suspicious": suspicious,
        "summary": summary,
    }

# --- RAG Setup (LangChain + FAISS) ---
def build_langchain_documents(sheet_a: pd.DataFrame, sheet_b: pd.DataFrame, reconciliation: dict) -> list[Document]:
    docs = []
    
    for row in sheet_a.to_dict("records"):
        text = f"{row['transaction_id']} from {row['sheet']} row {row['row_number']}: date {row['date']}, reference {row['reference']}, party {row['counterparty'] or row['description']}, amount {row['amount']}, direction {row['direction']}."
        docs.append(Document(page_content=text, metadata=row))
        
    for row in sheet_b.to_dict("records"):
        text = f"{row['transaction_id']} from {row['sheet']} row {row['row_number']}: date {row['date']}, reference {row['reference']}, party {row['counterparty'] or row['description']}, amount {row['amount']}, direction {row['direction']}."
        docs.append(Document(page_content=text, metadata=row))
        
    for kind in ["matched", "mismatched", "missing_in_a", "missing_in_b", "suspicious"]:
        for item in reconciliation[kind]:
            text = f"{kind} findings: {str(item)}"
            metadata = {"kind": kind}
            # Handle list attributes mapping issue in Langchain metadata by stringifying dictionary values
            metadata.update({k: str(v) for k, v in item.items()})
            docs.append(Document(page_content=text, metadata=metadata))
            
    summary_text = f"summary findings: {str(reconciliation['summary'])}"
    docs.append(Document(page_content=summary_text, metadata={"kind": "summary"}))
    return docs

def initialize_vector_store(api_key: str, docs: list[Document]):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def ask_rag_agent(question: str, vectorstore: FAISS, api_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    
    template = """
    You are a financial transaction reconciliation assistant.
    Answer using only the evidence below. Cite transaction IDs or row numbers whenever possible.
    If evidence is insufficient, say what needs manual review.
    
    Question: {question}
    Evidence: {context}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(question)


# --- Streamlit UI ---
def main():
    st.title("Financial Transaction Reconciliation System")
    st.markdown("Automated reconciliation and conversational querying using RAG (LangChain + FAISS) and Gemini.")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Google Gemini API Key", type="password", help="Get a free key from Google AI Studio")
        st.divider()
        st.header("Upload Files")
        file_a = st.file_uploader("Sheet A (Internal Ledger)", type=["csv", "xls", "xlsx"])
        file_b = st.file_uploader("Sheet B (External Statement)", type=["csv", "xls", "xlsx"])
        reconcile_btn = st.button("Reconcile Sheets", type="primary")
        
    if "reconciliation" not in st.session_state:
        st.session_state.reconciliation = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if reconcile_btn:
        if not file_a or not file_b:
            st.error("Please upload both sheets.")
            return
        if not api_key:
            st.error("Please provide a Gemini API Key to initialize the RAG embeddings.")
            return
            
        with st.spinner("Processing sheets..."):
            try:
                sheet_a = load_sheet(file_a, "Sheet A")
                sheet_b = load_sheet(file_b, "Sheet B")
                recon_results = reconcile(sheet_a, sheet_b)
                st.session_state.reconciliation = recon_results
                
                # Build RAG
                docs = build_langchain_documents(sheet_a, sheet_b, recon_results)
                vectorstore = initialize_vector_store(api_key, docs)
                st.session_state.vectorstore = vectorstore
                st.success("Reconciliation complete and RAG index built!")
            except Exception as e:
                st.error(f"Error during reconciliation: {e}")
                
    if st.session_state.reconciliation:
        recon = st.session_state.reconciliation
        summary = recon["summary"]
        
        st.subheader("Reconciliation Snapshot")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Matched", summary["matched_count"])
        col2.metric("Mismatched", summary["mismatched_count"])
        col3.metric("Missing in A", summary["missing_in_a_count"])
        col4.metric("Missing in B", summary["missing_in_b_count"])
        col5.metric("Net Variance", summary["net_variance"])
        
        st.divider()
        
        st.subheader("Agent Consultation")
        question = st.chat_input("Ask about missing, mismatched, or suspicious transactions...")
        
        if question:
            st.chat_message("user").write(question)
            if not st.session_state.vectorstore:
                st.error("Vector store not initialized.")
            elif not api_key:
                st.error("Gemini API key is required to query the agent.")
            else:
                with st.spinner("Consulting the agent..."):
                    try:
                        answer = ask_rag_agent(question, st.session_state.vectorstore, api_key)
                        st.chat_message("assistant").write(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
