"""
Lab 7 Demo — Embedding & Vector Store
Streamlit app: real embeddings (sentence-transformers) + Gemini chatbot.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=False)

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    ParentChildChunker,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
    compute_similarity,
)

# ── constants ─────────────────────────────────────────────────────────────────
DATASET_DIR = Path("data/dataset")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ALL_STRATEGIES = ["FixedSizeChunker", "SentenceChunker", "RecursiveChunker", "ParentChildChunker"]

RAG_SYSTEM_PROMPT = """Bạn là trợ lý tư vấn sản phẩm du lịch Vinpearl.
Trả lời dựa trên các đoạn thông tin được cung cấp bên dưới.
Nếu thông tin không có trong context, hãy nói rõ là bạn không có thông tin đó.
Trả lời bằng tiếng Việt, ngắn gọn và chính xác."""

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang tải model embedding (lần đầu ~30s)...")
def load_embedder():
    """Load sentence-transformers once, reuse across reruns."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed(text: str) -> list[float]:
            v = model.encode(text, normalize_embeddings=True)
            return v.tolist()
        return embed, "all-MiniLM-L6-v2 (local)"
    except Exception as e:
        return _mock_embed, f"mock (sentence-transformers unavailable: {e})"


def get_gemini_llm():
    """Return a Gemini generate function, or None if key missing."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your-gemini-key-here":
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        def generate(prompt: str) -> str:
            resp = model.generate_content(prompt)
            return resp.text
        return generate
    except Exception as e:
        st.warning(f"Gemini lỗi: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_dataset() -> dict[str, str]:
    files = {}
    if DATASET_DIR.exists():
        for p in sorted(DATASET_DIR.glob("*.md")):
            files[p.name] = p.read_text(encoding="utf-8")
    return files


def parse_meta(text: str, filename: str) -> dict:
    meta = {"filename": filename}
    for pattern, key in [
        (r"^# (.+)", "title"),
        (r"- URL: (.+)", "url"),
        (r"- Giá hiện tại: (.+)", "current_price"),
        (r"- Giá gốc: (.+)", "original_price"),
    ]:
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            meta[key] = m.group(1).strip()
    return meta


def make_chunker(name: str, chunk_size: int, overlap: int, max_sentences: int):
    if name == "FixedSizeChunker":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if name == "SentenceChunker":
        return SentenceChunker(max_sentences_per_chunk=max_sentences)
    if name == "RecursiveChunker":
        return RecursiveChunker(chunk_size=chunk_size)
    return ParentChildChunker(parent_chunk_size=chunk_size * 2, child_chunk_size=chunk_size)


def chunk_stats(chunks: list[str]) -> dict:
    if not chunks:
        return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
    lens = [len(c) for c in chunks]
    return {"count": len(lens), "avg_len": int(sum(lens)/len(lens)),
            "min_len": min(lens), "max_len": max(lens)}


@st.cache_resource(show_spinner="Đang build index...")
def build_index(file_names: tuple[str, ...], strategy: str,
                chunk_size: int, overlap: int, max_sentences: int) -> EmbeddingStore:
    dataset = load_dataset()
    embed_fn, _ = load_embedder()
    store = EmbeddingStore(collection_name="demo", embedding_fn=embed_fn)
    chunker = make_chunker(strategy, chunk_size, overlap, max_sentences)
    docs = []
    for fname in file_names:
        text = dataset[fname]
        meta = parse_meta(text, fname)
        if isinstance(chunker, ParentChildChunker):
            pairs = chunker.chunk_with_parents(text)
            for i, (parent, child) in enumerate(pairs):
                docs.append(Document(
                    id=f"{fname}_{i}",
                    content=child,
                    metadata={**meta, "chunk_index": i, "parent_content": parent},
                ))
        else:
            chunks = chunker.chunk(text)
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    id=f"{fname}_{i}",
                    content=chunk,
                    metadata={**meta, "chunk_index": i},
                ))
    store.add_documents(docs)
    return store


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lab 7 — Vinpearl RAG Demo",
    page_icon="🏖️",
    layout="wide",
)

st.title("🏖️ Lab 7 — Vinpearl RAG Demo")

dataset = load_dataset()
embed_fn, embed_name = load_embedder()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    st.caption(f"**Embedding:** {embed_name}")
    gemini_ok = bool(GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-key-here")
    st.caption(f"**LLM:** {'✅ Gemini ' + GEMINI_MODEL if gemini_ok else '⚠️ Mock (chưa set GEMINI_API_KEY)'}")

    st.divider()
    st.subheader("📁 Dataset (chọn 5 file cho nhóm)")
    selected_files = st.multiselect(
        "Files",
        options=list(dataset.keys()),
        default=list(dataset.keys())[:5],
        max_selections=10,
    )
    st.caption(f"Đang chọn {len(selected_files)} / {len(dataset)} file")

    st.divider()
    st.subheader("🔧 Tham số Chunker")
    chunk_size    = st.slider("chunk_size", 100, 1000, 400, 50)
    overlap       = st.slider("overlap (FixedSize)", 0, 200, 50, 10)
    max_sentences = st.slider("max_sentences (Sentence)", 1, 10, 3)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_chunk, tab_cosine, tab_export = st.tabs([
    "💬 Chatbot RAG",
    "📊 So sánh Chunking",
    "📐 Cosine Similarity",
    "📋 Export Report",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CHATBOT RAG
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not selected_files:
        st.info("Chọn ít nhất 1 file ở sidebar.")
        st.stop()

    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.subheader("Cấu hình RAG")
        rag_strategy = st.selectbox("Chunking strategy", ALL_STRATEGIES, index=2)
        top_k = st.slider("Top-k chunks", 1, 8, 3)
        show_sources = st.toggle("Hiển thị nguồn", value=True)
        if st.button("🔄 Rebuild Index", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        store = build_index(
            tuple(selected_files), rag_strategy,
            chunk_size, overlap, max_sentences
        )
        st.metric("Chunks trong index", store.get_collection_size())
        st.metric("Files đã index", len(selected_files))

    with col_left:
        st.subheader("Chat với Vinpearl Assistant")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                    with st.expander("📎 Nguồn tham khảo"):
                        for s in msg["sources"]:
                            st.caption(f"**{s['title']}** | score={s['score']:.4f}")
                            if s.get("parent"):
                                st.caption("🔍 Child:"); st.text(s["content"][:150] + "...")
                                st.caption("📄 Parent:"); st.text(s["parent"][:300] + "...")
                            else:
                                st.text(s["content"][:200] + "...")

        # Chat input
        if prompt := st.chat_input("Hỏi về sản phẩm Vinpearl..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Đang tìm kiếm và sinh câu trả lời..."):
                    # Retrieve
                    results = store.search(prompt, top_k=top_k)
                    context = "\n\n".join(
                        f"[{i+1}] (nguồn: {r['metadata'].get('title','?')})\n"
                        f"{r['metadata'].get('parent_content') or r['content']}"
                        for i, r in enumerate(results)
                    )
                    full_prompt = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nCâu hỏi: {prompt}\n\nTrả lời:"

                    llm = get_gemini_llm()
                    if llm:
                        answer = llm(full_prompt)
                    else:
                        answer = (
                            "⚠️ Chưa có Gemini API key. Thêm `GEMINI_API_KEY` vào file `.env`.\n\n"
                            "**Context tìm được:**\n" +
                            "\n".join(f"- {r['content'][:120]}..." for r in results)
                        )

                st.markdown(answer)
                sources = [{"title": r["metadata"].get("title","?"),
                            "score": r["score"], "content": r["content"],
                            "parent": r["metadata"].get("parent_content")}
                           for r in results]
                if show_sources:
                    with st.expander("📎 Nguồn tham khảo"):
                        for s in sources:
                            st.caption(f"**{s['title']}** | score={s['score']:.4f}")
                            if s["parent"]:
                                st.caption("🔍 Child (matched):")
                                st.text(s["content"][:150] + "...")
                                st.caption("📄 Parent (context sent to LLM):")
                                st.text(s["parent"][:300] + "...")
                            else:
                                st.text(s["content"][:200] + "...")

            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources
            })

        if st.session_state.messages:
            if st.button("🗑️ Xoá lịch sử chat"):
                st.session_state.messages = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — CHUNKING COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
with tab_chunk:
    st.subheader("So sánh Chunking Strategy")
    if not selected_files:
        st.info("Chọn file ở sidebar.")
        st.stop()

    # Summary table
    rows = []
    for fname in selected_files:
        text = dataset[fname]
        short = fname[:40]
        for name in ALL_STRATEGIES:
            chunker = make_chunker(name, chunk_size, overlap, max_sentences)
            chunks = chunker.chunk(text)
            s = chunk_stats(chunks)
            rows.append({"File": short, "Strategy": name,
                         "Chunks": s["count"], "Avg len": s["avg_len"],
                         "Min": s["min_len"], "Max": s["max_len"]})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Avg len": st.column_config.NumberColumn(format="%d"),
                                "Chunks": st.column_config.NumberColumn(format="%d")})

    st.divider()
    st.subheader("Xem chunk thực tế")
    c1, c2 = st.columns(2)
    with c1:
        pfile = st.selectbox("File", selected_files, key="pf")
    with c2:
        pstrat = st.selectbox("Strategy", ALL_STRATEGIES, key="ps")

    chunker = make_chunker(pstrat, chunk_size, overlap, max_sentences)
    pchunks = chunker.chunk(dataset[pfile])
    s = chunk_stats(pchunks)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Số chunks", s["count"])
    m2.metric("Avg length", s["avg_len"])
    m3.metric("Min length", s["min_len"])
    m4.metric("Max length", s["max_len"])

    idx = st.slider("Chunk #", 0, max(len(pchunks)-1,0), 0)
    if pchunks:
        st.text_area(f"Chunk {idx+1}/{len(pchunks)} ({len(pchunks[idx])} ký tự)",
                     pchunks[idx], height=200)

    if pstrat == "ParentChildChunker":
        st.divider()
        st.subheader("Parent ↔ Child")
        pc = ParentChildChunker(parent_chunk_size=chunk_size*2, child_chunk_size=chunk_size)
        pairs = pc.chunk_with_parents(dataset[pfile])
        if pairs:
            pi = st.slider("Cặp #", 0, len(pairs)-1, 0)
            parent_t, child_t = pairs[pi]
            l, r = st.columns(2)
            with l:
                st.markdown("**Parent** — context cho LLM")
                st.text_area("p", parent_t, height=180, label_visibility="collapsed")
            with r:
                st.markdown("**Child** — embed & search")
                st.text_area("c", child_t, height=180, label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════
with tab_cosine:
    st.subheader("Cosine Similarity Explorer")
    use_real = st.toggle("Dùng real embedding (sentence-transformers)", value=True)
    ef = embed_fn if use_real else _mock_embed
    st.caption(f"Đang dùng: **{'sentence-transformers' if use_real else 'mock (hash)'}**")

    c1, c2 = st.columns(2)
    with c1:
        sa = st.text_area("Sentence A", "Voucher nghỉ dưỡng tại Vinpearl Nha Trang.", height=100)
    with c2:
        sb = st.text_area("Sentence B", "Kỳ nghỉ 3 ngày 2 đêm tại khách sạn biển Nha Trang.", height=100)

    if st.button("Tính similarity", type="primary"):
        score = compute_similarity(ef(sa), ef(sb))
        color = "green" if score > 0.5 else ("orange" if score > 0.2 else "red")
        st.markdown(f"### Cosine Similarity: :{color}[{score:.4f}]")
        st.progress(float(max(0, (score+1)/2)))
        if score > 0.7:
            st.success("HIGH similarity")
        elif score > 0.3:
            st.warning("MEDIUM similarity")
        else:
            st.error("LOW similarity")

    st.divider()
    st.subheader("Bulk — Sec 5 Report")
    default_pairs = [
        ("Voucher nghỉ dưỡng Vinpearl Nha Trang.", "Kỳ nghỉ biển tại khách sạn Nha Trang."),
        ("Chính sách hoàn hủy vé.", "Thời tiết hôm nay đẹp."),
        ("Vé vào cửa VinWonders Phú Quốc.", "Vé vào cửa VinWonders Phú Quốc."),
        ("Bao gồm bữa sáng cho 2 người.", "Ăn sáng được phục vụ cho cặp đôi."),
        ("Phụ thu cuối tuần 600.000đ.", "Sân golf Vinpearl Phú Quốc."),
    ]
    bulk_rows = []
    for i, (a, b) in enumerate(default_pairs):
        score = compute_similarity(ef(a), ef(b))
        bulk_rows.append({"#": i+1, "A": a[:45], "B": b[:45],
                          "Score": round(score, 4),
                          "Dự đoán": "high" if score > 0.5 else "low",
                          "Actual": "high" if score > 0.3 else "low"})
    st.dataframe(pd.DataFrame(bulk_rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT REPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("Export số liệu cho REPORT.md")
    if not selected_files:
        st.info("Chọn file ở sidebar.")
        st.stop()

    # Baseline table
    st.markdown("#### Baseline Analysis (Sec 3)")
    comp = ChunkingStrategyComparator()
    rrows = []
    for fname in selected_files:
        text = dataset[fname]
        short = fname[:38]
        for s_name, stats in comp.compare(text, chunk_size=chunk_size).items():
            rrows.append({"Tài liệu": short, "Strategy": s_name,
                          "Count": stats["count"], "Avg Len": int(stats["avg_length"])})
        pc = ParentChildChunker(parent_chunk_size=chunk_size*2, child_chunk_size=chunk_size)
        ch = pc.chunk(text)
        rrows.append({"Tài liệu": short, "Strategy": "ParentChildChunker",
                      "Count": len(ch), "Avg Len": sum(len(c) for c in ch)//max(len(ch),1)})

    df_r = pd.DataFrame(rrows)
    st.dataframe(df_r, use_container_width=True, hide_index=True)
    st.download_button("⬇️ CSV", df_r.to_csv(index=False).encode("utf-8"),
                       "baseline.csv", "text/csv")

    st.markdown("**Markdown (paste vào REPORT.md):**")
    md = ["| Tài liệu | Strategy | Chunk Count | Avg Length |",
          "|----------|----------|-------------|------------|"]
    for _, row in df_r.iterrows():
        md.append(f"| {row['Tài liệu']} | {row['Strategy']} | {row['Count']} | {row['Avg Len']} |")
    st.code("\n".join(md), language="markdown")

    st.divider()

    # Data inventory
    st.markdown("#### Data Inventory (Sec 2)")
    inv = []
    for fname in selected_files:
        text = dataset[fname]
        m = parse_meta(text, fname)
        inv.append({"Tên tài liệu": m.get("title", fname)[:50],
                    "Nguồn": "booking.vinpearl.com",
                    "Số ký tự": len(text),
                    "Giá hiện tại": m.get("current_price", "—"),
                    "Metadata gán": "title, url, current_price, chunk_index"})
    df_inv = pd.DataFrame(inv)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)
    st.download_button("⬇️ CSV inventory", df_inv.to_csv(index=False).encode("utf-8"),
                       "data_inventory.csv", "text/csv")

    st.divider()

    # Benchmark query runner
    st.markdown("#### Benchmark Queries (Sec 6)")
    st.caption("Nhập 5 query của nhóm, chạy và xem kết quả để điền bảng Sec 6.")
    default_queries = [
        "Voucher có hoàn hủy không?",
        "Bao gồm những dịch vụ gì?",
        "Giá hiện tại là bao nhiêu?",
        "Thời hạn sử dụng voucher đến khi nào?",
        "Có phụ thu cuối tuần không?",
    ]
    queries = []
    for i in range(5):
        q = st.text_input(f"Query {i+1}", default_queries[i], key=f"bq{i}")
        queries.append(q)

    if st.button("▶️ Chạy 5 Benchmark Queries", type="primary"):
        store_bm = build_index(tuple(selected_files), "ParentChildChunker",
                               chunk_size, overlap, max_sentences)
        bm_rows = []
        for i, q in enumerate(queries):
            if not q.strip():
                continue
            results = store_bm.search(q, top_k=3)
            top1 = results[0] if results else {}
            bm_rows.append({
                "#": i+1, "Query": q,
                "Top-1 chunk (tóm tắt)": top1.get("content","")[:80]+"..." if top1 else "",
                "Score": round(top1.get("score",0), 4) if top1 else 0,
                "Nguồn": top1.get("metadata",{}).get("title","")[:30] if top1 else "",
            })
        st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True)
