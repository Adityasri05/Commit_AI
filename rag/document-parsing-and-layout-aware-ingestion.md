# Document Parsing and Layout-Aware Ingestion for RAG

Retrieval-Augmented Generation begins before embeddings and vector search: the source documents must be parsed into useful, traceable content. A parser that drops table structure, mixes columns, loses headings, or merges unrelated sections can make retrieval fail even when the embedding model is strong.

Layout-aware ingestion preserves document structure and provenance so that chunks remain understandable after extraction.

## Why plain text extraction can fail

PDFs and office documents are presentation formats, not necessarily clean reading-order text. Common problems include:

- Two-column pages extracted in the wrong order.
- Headings separated from the paragraphs they introduce.
- Tables flattened into strings with ambiguous row and column relationships.
- Headers, footers, page numbers, and repeated navigation mixed into content.
- Scanned pages containing images rather than embedded text.
- Footnotes, captions, and references detached from the relevant passage.

A retrieval system cannot reliably recover structure that ingestion discarded.

## A robust ingestion pipeline

### 1. Identify the document type

Detect PDF, DOCX, HTML, slides, spreadsheets, images, and plain text. Use parsers appropriate to each format rather than sending every file through one generic extractor.

### 2. Extract layout elements

Represent content as typed elements where possible:

- Title and section heading.
- Paragraph and list item.
- Table, row, and cell.
- Figure, caption, and alt text.
- Header, footer, footnote, and page number.

Preserve coordinates or structural relationships when they help reconstruct reading order.

### 3. Apply OCR only when needed

Scanned PDFs and image-only pages need OCR. Detect pages with little or no embedded text and route those pages through OCR. Retain the original image or page reference so extraction can be reviewed.

OCR can confuse similar characters, omit small text, and struggle with tables or handwriting. Do not silently treat OCR output as ground truth.

### 4. Normalize without destroying meaning

Normalize whitespace, remove repeated boilerplate when safe, and preserve meaningful punctuation, units, symbols, code blocks, and list relationships. Keep the raw extraction alongside the normalized representation for debugging.

### 5. Chunk with document structure

Prefer boundaries such as headings, paragraphs, list groups, and table sections. A useful chunk should carry enough context to stand alone.

For a table, include its title, column headers, units, and relevant row labels. If a table is too large, split by row groups while repeating the necessary headers.

### 6. Attach provenance metadata

Every chunk should retain enough metadata to locate and verify its source:

- Stable document ID and document version.
- Source URI or storage key.
- Page number or slide number.
- Section heading path.
- Element type and extraction method.
- Bounding box or element identifier when available.
- Ingestion timestamp and parser version.

Do not expose internal storage paths or sensitive metadata to end users unless authorized.

## A PyMuPDF text-block baseline

This example extracts positioned text blocks from a text-based PDF. Sorting by coordinates is only a heuristic; complex columns, tables, and mixed layouts require layout-aware parsing and manual inspection.

```python
import fitz  # pip install pymupdf

def extract_blocks(pdf_path):
    records = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            for block in page.get_text("blocks", sort=True):
                x0, y0, x1, y1, text, block_no, block_type = block
                cleaned = text.strip()
                if cleaned:
                    records.append({
                        "page": page_number,
                        "block_id": block_no,
                        "block_type": block_type,
                        "bbox": [x0, y0, x1, y1],
                        "text": cleaned,
                    })
    return records

blocks = extract_blocks("handbook.pdf")
for item in blocks[:3]:
    print(item["page"], item["bbox"], item["text"][:120])
```

For table-heavy PDFs, evaluate a dedicated table extractor. For scanned documents, use OCR and retain confidence or review metadata. Libraries such as Docling and Unstructured can provide richer document-element representations, but their output still needs validation.

## Quality checks before indexing

- Compare extracted text with a sample of original pages.
- Check page coverage and detect empty or suspiciously short pages.
- Detect duplicate headers and repeated boilerplate.
- Validate table row/column counts and numeric values.
- Confirm heading paths and chunk boundaries.
- Record parser errors and OCR confidence where available.
- Re-run a small retrieval test using questions whose answers are in tables, captions, and multi-column sections.

## Common mistakes

- Assuming PDF text order equals visual reading order.
- Flattening a table without its headers and units.
- Removing all repeated text even when it is a section label or legal qualifier.
- Losing page-level provenance during chunking.
- Treating OCR output as verified source content.
- Re-indexing a changed document without retiring stale chunks from its previous version.

## Practice

Choose a PDF containing a two-column page, a table, a figure caption, and at least one scanned page. Compare basic text extraction with a layout-aware parser. Build five questions whose answers depend on different element types, then inspect whether retrieved chunks preserve the required context and source page.

## References

- [PyMuPDF documentation](https://pymupdf.readthedocs.io/)
- [Docling documentation](https://docling-project.github.io/docling/)
- [Unstructured documentation](https://docs.unstructured.io/)
- [OCRmyPDF documentation](https://ocrmypdf.readthedocs.io/)

**Key takeaway:** High-quality RAG starts with faithful extraction. Preserve layout, structure, and provenance before chunking or embedding anything.
