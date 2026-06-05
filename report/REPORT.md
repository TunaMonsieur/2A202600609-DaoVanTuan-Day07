# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Đào Văn Tuấn
**Nhóm:** [Tên nhóm]
**Ngày:** 2026-06-05

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có góc giữa chúng nhỏ, tức là chúng "chỉ cùng hướng" trong không gian embedding. Điều này có nghĩa là hai văn bản có nội dung / ngữ nghĩa tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Python is a popular programming language."
- Sentence B: "Python is widely used for software development."
- Tại sao tương đồng: Cả hai đều nói về Python trong bối cảnh lập trình — chủ thể, từ ngữ chuyên ngành và ý nghĩa gần giống nhau.

**Ví dụ LOW similarity:**
- Sentence A: "The weather is sunny and warm today."
- Sentence B: "Vector databases store high-dimensional embeddings."
- Tại sao khác: Hai câu hoàn toàn khác nhau về chủ đề (thời tiết vs. kỹ thuật CSDL), không chia sẻ từ vựng hay ngữ nghĩa chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến góc (hướng) giữa hai vector, không bị ảnh hưởng bởi độ dài vector — rất phù hợp vì các embedding thường được normalize về độ dài 1. Euclidean distance bị ảnh hưởng bởi magnitude, dẫn đến kết quả sai khi so sánh văn bản dài vs. ngắn dù cùng chủ đề.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính:
> - step = chunk_size − overlap = 500 − 50 = 450
> - Số chunks = ⌈(10000 − 500) / 450⌉ + 1 = ⌈9500 / 450⌉ + 1 = 22 + 1 = **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100: step = 400 → số chunks = ⌈9500 / 400⌉ + 1 = 24 + 1 = **25 chunks** (tăng thêm 2). Overlap lớn hơn giúp mỗi chunk giữ được context từ chunk liền kề, tránh bị "cắt đứt" thông tin quan trọng ở biên.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Aquafield Nha Trang | FixedSizeChunker (`fixed_size`) | 20 | 398 | Không — cắt giữa câu/bullet |
| Aquafield Nha Trang | SentenceChunker (`by_sentences`) | 15 | 466 | Một phần — gom bullet thành 1 chunk to |
| Aquafield Nha Trang | RecursiveChunker (`recursive`) | 27 | 258 | Tốt hơn — tách theo paragraph |
| Grand World | FixedSizeChunker (`fixed_size`) | 30 | 389 | Không — cắt giữa câu/bullet |
| Grand World | SentenceChunker (`by_sentences`) | 20 | 509 | Kém — chunk quá to do ít dấu `.` |
| Grand World | RecursiveChunker (`recursive`) | 42 | 242 | Tốt — giữ paragraph |

### Strategy Của Tôi

**Loại:** ParentChildChunker (custom — hai tầng)

**Mô tả cách hoạt động:**
> `ParentChildChunker` tách văn bản thành 2 tầng. **Parent chunk** (≤800 ký tự) dùng `RecursiveChunker` với separator `["\n## ", "\n\n", "\n"]` — tức là ưu tiên giữ nguyên từng section `##` làm một khối ngữ nghĩa lớn. **Child chunk** (≤200 ký tự) tách tiếp mỗi parent bằng `["\n\n", "\n- ", "\n+ ", "\n", ". ", " ", ""]` — chia nhỏ đến từng bullet hoặc câu. Khi indexing, **child được embed để search** (ngắn → vector chính xác hơn), nhưng **parent được lưu vào metadata** để LLM có đủ context khi generate.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Data Vinpearl có cấu trúc hai lớp tự nhiên: section lớn (`## Bao gồm`, `## Điều khoản`) chứa nhiều bullet nhỏ. Nếu embed cả section → vector bị "loãng" ngữ nghĩa. Nếu chỉ embed từng bullet riêng lẻ → LLM thiếu context khi trả lời. Parent-child giải quyết cả hai: **search bằng child** (precision cao), **answer bằng parent** (context đủ).

**Code snippet (nếu custom):**
```python
from src import ParentChildChunker
from src.models import Document

chunker = ParentChildChunker(parent_chunk_size=800, child_chunk_size=200)
pairs = chunker.chunk_with_parents(document_text)

# Index: embed child, giữ parent trong metadata
docs = [
    Document(
        id=f"{doc_id}_chunk_{i}",
        content=child,                          # embed cái này
        metadata={**base_meta, "parent_content": parent}  # context cho LLM
    )
    for i, (parent, child) in enumerate(pairs)
]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Aquafield Nha Trang | RecursiveChunker (best baseline) | 27 | 258 | Tốt — tách đúng paragraph |
| Aquafield Nha Trang | **ParentChildChunker (của tôi)** | **62** | **111** | **Tốt nhất — child nhỏ, vector chính xác; parent giữ context** |
| Grand World | RecursiveChunker (best baseline) | 42 | 242 | Tốt — tách đúng paragraph |
| Grand World | **ParentChildChunker (của tôi)** | **81** | **124** | **Tốt nhất — retrieval precision cao hơn nhờ child ngắn** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])\s+|(?<=\.)\n` với positive lookbehind để tách câu tại các dấu kết thúc câu (`.`, `!`, `?`) mà không xóa dấu câu đó khỏi kết quả. Edge case xử lý: văn bản trống trả về `[]`; nếu regex không tách được gì (không có dấu câu), toàn bộ text được trả về như một chunk duy nhất. Các câu được gom theo `max_sentences_per_chunk` rồi join bằng dấu cách.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Base case: nếu `len(current_text) <= chunk_size`, trả về nguyên văn. Algorithm thử lần lượt các separator từ ưu tiên cao đến thấp (`\n\n → \n → ". " → " " → ""`). Với mỗi separator, tách text và gom phần liên tiếp bằng thuật toán greedy: thêm phần tiếp vào `current` nếu vẫn ≤ chunk_size, ngược lại flush và đệ quy phần quá dài với `remaining_separators`. Separator `""` là fallback cuối cùng, cắt theo ký tự.

**`ParentChildChunker.chunk` / `chunk_with_parents`** *(strategy cá nhân tôi chọn)* — approach:
> Wrap 2 `RecursiveChunker` lồng nhau: parent chunker (chunk_size=800, separators `["\n## ", "\n\n", "\n"]`) tạo ra các khối section lớn; child chunker (chunk_size=200, separators `["\n\n", "\n- ", "\n+ ", "\n", ". ", " ", ""]`) tách tiếp mỗi parent. `chunk()` trả về list child strings (tương thích interface cũ). `chunk_with_parents()` trả về list `(parent, child)` để indexing: child được embed cho search, parent được lưu vào `metadata["parent_content"]` để LLM có đủ context.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` gọi `_embedding_fn(doc.content)` cho mỗi document rồi lưu dict `{id, content, embedding, metadata}` vào `self._store` (hoặc ChromaDB nếu có). `search` embed câu query, tính **dot product** giữa query vector và từng stored embedding (do embeddings đã được normalize về đơn vị 1 nên dot product = cosine similarity), sắp xếp descending và trả về top_k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` **filter trước**: lọc `self._store` giữ lại chỉ những record có tất cả key-value trong `metadata_filter` khớp, sau đó mới chạy similarity search trên tập con đó. `delete_document` rebuild lại list `self._store` bằng list comprehension loại bỏ mọi record có `metadata['doc_id'] == doc_id`, trả về `True` nếu store có shrink.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top_k chunk từ store cho câu hỏi. Format mỗi chunk thành `[i] content` (numbered để LLM có thể tham chiếu). Inject context vào prompt theo cấu trúc: `Context:\n{chunks}\n\nQuestion: {question}\n\nAnswer:`. Toàn bộ prompt được truyền vào `llm_fn` và kết quả trả về trực tiếp — không có post-processing.

### Test Results

```
# Chạy: python -m pytest tests/ -v
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

> **Embedding:** all-MiniLM-L6-v2 (sentence-transformers) — scores phản ánh ngữ nghĩa thực.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Python is a programming language." | "Python is widely used for coding." | high | 0.7757 | Có |
| 2 | "The weather is sunny today." | "Vector stores hold embeddings." | low | -0.0807 | Có |
| 3 | "Hello world" | "Hello world" | high (=1.0) | 1.0000 | Có |
| 4 | "Machine learning uses data." | "Deep learning is a subset of ML." | high | 0.5137 | Có |
| 5 | "I love pizza." | "Quantum physics is complex." | low | 0.0878 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 4 bất ngờ nhất — "Machine learning uses data" và "Deep learning is a subset of ML" chỉ đạt 0.51, không cao hơn nhiều dù cùng lĩnh vực. Điều này cho thấy `all-MiniLM-L6-v2` encode **surface form** (từ ngữ cụ thể) quan trọng hơn **concept-level similarity** — hai câu dùng từ khác nhau hoàn toàn nên vector không gần nhau bằng pair 1 (cùng dùng "Python"). Đây là hạn chế của embedding nhỏ so với LLM lớn.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Giá vé hiện tại của VinWonders Nha Trang là bao nhiêu? | Giá hiện tại **500.000 đ** (giá gốc 600.000 đ), theo trang VinWonders Nha Trang. |
| 2 | Aquafield Nha Trang có những phòng trị liệu xông hơi nào? | Có 7 phòng: Băng tuyết, Gỗ bách (Hinoki), Sương mây, Đá muối Himalaya, Bulgama, Than củi, Hoàng thổ — mỗi phòng có nhiệt độ/độ ẩm riêng. |
| 3 | Combo 2N1Đ Vinpearl Hotel Cần Thơ bao gồm những dịch vụ gì? | 01 đêm phòng Deluxe (2 người lớn + 2 trẻ dưới 4 tuổi), 01 bữa sáng, miễn phụ thu cuối tuần, thuế phí dịch vụ. |
| 4 | Voucher golf Sunrise áp dụng tee-time và ngày nào? | Sunrise: tee-time **trước 8:00**, nhóm từ 2 khách, **thứ 2–thứ 6**, không áp dụng ngày lễ 30/4, 01/05, 02/9. |
| 5 | Night Safari tại Vinpearl Safari Phú Quốc là gì? | Hành trình khám phá động vật về đêm bằng xe điện — trải nghiệm safari ban đêm duy nhất tại Việt Nam. |

### Kết Quả Của Tôi

> **Cấu hình:** ParentChildChunker (parent=800, child=200), Embedding: all-MiniLM-L6-v2, 10 files / 512 chunks

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Giá vé VinWonders Nha Trang? | "Tại sao nên đến VinWonders Nha Trang?" — giới thiệu chung | 0.7218 | Có | Trả về thông tin tổng quan VinWonders NT |
| 2 | Các phòng trị liệu Aquafield NT? | "Các phòng trị liệu tại Aquafield Nha Trang" — header section | 0.8094 | Có | Liệt kê đúng 7 phòng xông hơi |
| 3 | Dịch vụ Combo Cần Thơ 2N1Đ? | "1. Các dịch vụ bao gồm — 01 đêm phòng Deluxe..." | 0.7276 | Có | Mô tả đúng phòng + bữa sáng + miễn phụ thu |
| 4 | Voucher golf Sunrise? | "Tee-time trước 8:00 — thứ 2 đến thứ 6..." | 0.6605 | Có | Trả lời đúng thời gian và ngày áp dụng |
| 5 | Night Safari Vinpearl Safari? | "Khám Phá Thế Giới Hoang Dã tại Vinpearl Safari Phú Quốc" | 0.6913 | Có | Mô tả safari ban đêm bằng xe điện |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **5 / 5**

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
