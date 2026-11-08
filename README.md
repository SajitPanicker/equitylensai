# Equity Lens

An exchange-connected Streamlit research application for discovering company disclosures from official NSE India and BSE India sources, previewing selected records, downloading selected PDF/PPTX files into an application-managed cache, and indexing them in the existing persistent ChromaDB collection. A parent LangGraph classifies each question and routes it to exchange-filing RAG, Yahoo Finance, or both. The app retains `sentence-transformers/all-mpnet-base-v2` embeddings, recursive chunking, cosine retrieval, and OpenAI `gpt-4o-mini` answers.

Filing answers use indexed evidence. Market answers use Yahoo Finance's latest available data, which is not guaranteed real-time. Filing search is explicit and bounded; no filings are downloaded until selected. The app is not investment advice.

## Requirements

- Windows 10 or 11
- Python 3.11 or 3.12
- An OpenAI API key and an account with API access to the configured model
- Internet access for package/model downloads and OpenAI requests

## Windows setup

### 1. Install and check Python

Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/windows/). In the installer, enable **Add python.exe to PATH**. Open a new PowerShell or Command Prompt window and check:

```powershell
python --version
```

The output should begin with `Python 3.11` or `Python 3.12`. If `python` is not found, rerun the installer with PATH enabled or use the Python launcher (`py -3.12`).

### 2. Create the project folder

Open PowerShell and run:

```powershell
New-Item -ItemType Directory -Force "$HOME\Desktop\Equity Lens"
```

Keep the project files in the workspace. Exchange downloads and search-result cache are created automatically under `EXCHANGE_CACHE_PATH`; users do not maintain a filing upload folder.

### 3. Configure `.env`

Copy `.env.example` to `.env` in the project folder. Replace the API key placeholder with your own key:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
EXCHANGE_CACHE_PATH=./exchange_cache
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
OPENAI_MODEL=gpt-4o-mini
IMAGE_MODEL=gpt-image-2.5-flare
IMAGE_SIZE=1024x1536
IMAGE_QUALITY=medium
TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

Do not add quotes around ordinary values. Keep the key private: `.env` is excluded by `.gitignore`; never commit it, paste it into source code, or share it. `EXCHANGE_CACHE_PATH` and `CHROMA_DB_PATH` are persistent local application data, not upload locations.

### 4. Create and activate a virtual environment

In PowerShell:

```powershell
cd "$HOME\Desktop\Equity Lens"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead, or temporarily allow local activation scripts for this window with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

In Command Prompt (`cmd.exe`):

```bat
cd /d "%USERPROFILE%\Desktop\Equity Lens"
python -m venv .venv
.venv\Scripts\activate.bat
```

The prompt should show `(.venv)` after activation.

### 5. Upgrade pip and install packages

With the virtual environment active, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists the direct third-party packages used by this project. pip resolves and installs their transitive dependencies automatically. The sentence-transformers installation includes the PyTorch and model-loading dependencies it needs; the first embedding-model load downloads model files and can take a few minutes. Pillow is included for compositing exact, verified source quotations over generated image backgrounds. `langchain-text-splitters` provides the recursive splitter only; retrieval and generation continue to use ChromaDB and the OpenAI SDK directly.

### 6. Start the Streamlit server

This project is a Streamlit application. Streamlit starts and manages its own Uvicorn-based web server, so you do not start it with `uvicorn stock_rag_chatbot:app`; this script does not expose a FastAPI/ASGI `app` object. From the project folder, with the virtual environment active, run:

```powershell
python -m streamlit run stock_rag_chatbot.py
```

This command starts Streamlit and opens the local app in your browser (normally at `http://localhost:8501`). Choose an exchange and exact identifier, select a financial-year or publication-date range and categories, search, preview, then explicitly select filings to download and index. Keep the terminal open while using the app; press `Ctrl+C` there to stop the server.

## Exchange discovery and RAG pipeline

1. The primary interface searches by exact NSE symbol and/or numeric BSE scrip code. An optional company-name field is an exact legal-name filter; company names alone are not used to resolve identifiers.
2. Search periods are latest (past 365 days), selected financial years, or a custom publication-date range. NSE's filing API and BSE's search page provide exchange publication dates; fiscal year and quarter are stored only when stated in the filing text. The exchange may restrict historical results, and the app caps each exchange result set at 100 records.
3. Category filters cover financial results, annual reports, investor presentations, material announcements, earnings calls/analyst meetings, shareholding patterns, governance/compliance, credit ratings, board meetings, and other company filings. Original exchange title and category are retained. Keyword filtering is optional.
4. Search results are previewed and paginated before any download. Users explicitly select records, download selected records, then index them. **Refresh this result page** is an explicit group action; it does not fetch every year or every company announcement.
5. The exchange adapter validates HTTPS hostnames, every redirect, response status/content type, PDF/PPTX signatures, and a 50 MB limit. Requests have timeouts, bounded retries, exponential backoff, and per-host throttling. Files and search results are cached under `EXCHANGE_CACHE_PATH` by exchange/company/period/category.
6. PDFs are extracted page by page; PPTX presentations are extracted slide by slide, including table text. Empty or unreadable documents are reported as failed. Scanned PDFs require OCR before their text can be indexed; OCR is not silently claimed.
7. `RecursiveCharacterTextSplitter` keeps page/slide boundaries, using `CHUNK_SIZE=1000` and `CHUNK_OVERLAP=150` by default. `sentence-transformers/all-mpnet-base-v2` generates normalized 768-dimensional vectors.
8. The existing persistent ChromaDB collection `investor_presentations` remains shared across companies and years, with cosine similarity. Chunk metadata includes available company, ticker/code, exchange, financial year, quarter, reporting period, original category/title, publication date, source URL/path, SHA-256 hash, and page/slide location. Missing fields remain empty rather than being inferred.
9. Content hashes prevent duplicate embeddings within a selected batch and unchanged downloads reuse their cache file. Re-indexing a changed filing replaces its prior chunks safely. When identical NSE/BSE files are selected together, one copy is embedded and both official references are saved in metadata.
10. Retrieval applies company, exchange, category, year, quarter, and publication-date metadata filters before semantic search. Multi-year requests retrieve each selected fiscal year independently. GPT answers distinguish reported results, material announcements, and management commentary, and cite filing title, exchange, publication date, official URL, and page/slide where available.

## Question routing and graph architecture

The Streamlit chat invokes the parent graph in `equitylens_graph.py`. Its `intent_classification_node` calls the reusable OpenAI classification helper in `stock_rag_chatbot.py`, then validates the returned source flags. `route_by_intent` is ordinary Python: LangGraph's `add_conditional_edges()` uses its `"rag"`, `"yahoo"`, or `"both"` result to choose the next node.

The parent invokes the question-time RAG subgraph in `rag_graph.py` to return retrieved filing evidence and the Yahoo Finance subgraph in `yahoo_finance_graph.py` to return structured, bounded market data. For financial questions, `financial_synthesis_node` sends the original question, recent conversation history, and only the selected source data to GPT-4o-mini for a grounded analysis. Yahoo Finance historical returns and available revenue/profit changes are calculated in Python and passed separately from source figures. If both sources are selected, LangGraph runs them in parallel and waits for both before synthesis; a failed source is disclosed while available evidence can still be used. Explicit image requests retain the existing image-generation path. Conversation messages are checkpointed in `equitylens_checkpoints.sqlite`, with a separate thread ID for each Streamlit session. This routing does not change filing ingestion, the ChromaDB collection, or the embedding model.

Yahoo Finance is accessed through `yfinance`, an unofficial interface to Yahoo Finance. Its data can be delayed, incomplete, or unavailable; it must not be represented as guaranteed real-time exchange data. Review Yahoo/yfinance terms and applicable commercial-use and redistribution requirements before deploying or redistributing the app.

### Verified exchange access and limitations

- **NSE India announcements:** the official [Corporate Filings - Announcements](https://www.nseindia.com/companies-listing/corporate-filings-announcements) page calls `GET https://www.nseindia.com/api/corporate-announcements?index=equities...`. The live request returned HTTP 200 JSON with `symbol`, `sm_name`, `desc`, `attchmntText`, `an_dt`, `attchmntFile`, `fileSize`, and `seq_id` fields.
- **NSE annual reports:** the official [Annual Reports](https://www.nseindia.com/companies-listing/corporate-filings-annual-reports) page calls `GET https://www.nseindia.com/api/annual-reports?index=equities&symbol=INFY`. The live response returned company name, `fromYr`/`toYr`, submission type, publication timestamps, file size, and an official NSE Archives PDF URL.
- **NSE financial results:** the official [Financial Results](https://www.nseindia.com/companies-listing/corporate-filings-financial-results) page calls `GET https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol=INFY&period=Quarterly` (and `Half-Yearly`/`Annual`). The live response included reporting dates, `financialYear`, `relatingTo`, audited/consolidated basis, and XBRL/detail links. These entries are previewable; XBRL/XML and HTML files are not treated as indexable PDFs. The announcements feed also includes downloadable result attachments when the exchange provides them.
- For [shareholding patterns](https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern), [governance](https://www.nseindia.com/companies-listing/corporate-filings-governance), and [board meetings](https://www.nseindia.com/companies-listing/corporate-filings-board-meetings), the app searches the official equities announcement feed and exposes the dedicated official search page as a fallback. It does not claim that every specialized catalogue is completely mirrored by that feed.
- **BSE India:** the official search page is [Latest Corporate Announcements](https://www.bseindia.com/corporates/ann). Its live page issued a request to `https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w` with date, category, and scrip parameters. A direct request from this environment returned HTTP 403, so automatic BSE retrieval is not claimed as working here; the app displays the official search link. The BSE search page exposes announcement categories and paging, and separate official results and board-meeting pages.
- Official exchange feeds may not expose every filing as a downloadable PDF/PPTX. XBRL, HTML, transcript audio, and other unsupported formats can be opened at the official source but are not indexed by the current extractors. Legacy `.ppt` binary files are not supported; PDF and `.pptx` are supported.
- No CAPTCHA, access control, or exchange denial is bypassed. Search history and request quotas are controlled by the exchange; narrow dates and categories for older or large searches. The local 100-record cap is a performance guard, not a claim that the exchange has no more records.

The NSE announcement response does not return a separate formal title field: the UI retains its attachment-description text as the displayed title and preserves `desc` as the original exchange category. The annual-report API also has no formal title field; the UI labels the filing from its explicit year range and submission type. The official NSE company metadata endpoint `https://www.nseindia.com/api/equity-metainfo?symbol=INFY` was verified to resolve `INFY` to Infosys Limited and return an empty object for an unknown symbol.

## Image generation

- Ordinary questions such as “What was revenue in FY2025?” stay on the text-only RAG path. The app does not make an image request for them.
- Only a clear request to create, generate, draw, or produce a visual output is routed to image generation. A question such as “What information would a revenue infographic contain?” stays text-only. Tentative visual ideas can prompt a short clarification.
- The app retrieves presentation excerpts first. GPT-4o-mini prepares a structured image brief with exact quotations and source indices. The app checks each quotation against the retrieved text and only overlays quotations that match; the image model is asked to create a decorative background without factual labels or figures.
- Requests for a **graph**, **chart**, or **plot** use a stricter path: GPT-4o-mini proposes dated points and verbatim source quotes, then the app validates each period, value, metric, and unit against retrieved excerpts. Pillow draws the graph locally; the image model is not used to draw data. A graph needs at least two individually dated observations of the same metric and unit. A multi-year CAGR is one summary value, not a sequence of annual points. If the retrieved excerpts do not contain enough comparable values, the app explains that and refuses to invent the graph. Increasing `TOP_K` or asking about a specific presentation section can help retrieve more relevant data, but cannot supply figures absent from the documents.
- `client.images.generate()` uses the configurable `IMAGE_MODEL`. The default is `gpt-image-2.5-flare`; the official image-generation guide also documents `gpt-image-2.5-sunburst`. Portrait `1024x1536` is a documented supported size for an A4-style layout; this is a pixel aspect ratio, not print-ready A4 resolution.
- The app uses Pillow to overlay exact, source-checked text and citations, displays the PNG in chat, and provides a download button. This avoids relying on the image model to spell financial names, dates, or figures correctly.
- Image generation is a separate paid API capability. Verify that your OpenAI organization and project have access to the selected image model. Organization verification may be required for GPT Image models. `gpt-4o-mini` access for text does not by itself guarantee image-model access.

### Test the routing

With the virtual environment active, run the local tests:

```powershell
python -m unittest discover -s tests -v
```

Optional live smoke checks are separate from the automated suite and make requests only to the official exchange pages. The BSE check may report its current HTTP 403 restriction and print the official fallback link:

```powershell
python tests/manual_exchange_smoke.py --exchange NSE --nse-symbol INFY
python tests/manual_exchange_smoke.py --exchange BSE --bse-code 500209
```

The tests use mocked exchange responses, ChromaDB results, and OpenAI calls; they do not depend on exchange availability or spend API credits. Search and index filings in the app, then try these prompts:

| Prompt | Expected behavior |
| --- | --- |
| `What was revenue in FY2025?` | Text answer only |
| `Summarize the latest quarterly results` | Text answer only, grounded in indexed filings |
| `Create an A4 infographic showing revenue growth` | Retrieve evidence, generate and display a PNG infographic |
| `Create a graph of revenue growth by fiscal year` | Draw a graph only if at least two comparable, source-verified fiscal-year values are retrieved; otherwise explain which data is missing |
| `Generate an infographic about the company's business segments` | Retrieve evidence, generate and display a PNG infographic |
| `What information would a revenue infographic contain?` | Text answer only |

Also check the missing-key and simulated image-API failure tests in the command above; they verify the app returns useful errors instead of crashing.

Scanned PDFs that contain page images but no selectable text are not OCR'd by this app. If a PDF has no extractable text, the app reports it as a file-level indexing error.

## Configuration

| Setting | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Secret key used for intent classification and answer generation through OpenAI |
| `EXCHANGE_CACHE_PATH` | Persistent app-managed filing and search-result cache; defaults to `./exchange_cache` |
| `CHROMA_DB_PATH` | Persistent local ChromaDB data directory |
| `EMBEDDING_MODEL` | Hugging Face sentence-transformer used for both documents and questions |
| `OPENAI_MODEL` | OpenAI chat model; defaults to `gpt-4o-mini` |
| `IMAGE_MODEL` | OpenAI Images API model; defaults to `gpt-image-2.5-flare` |
| `IMAGE_SIZE` | Supported output size; defaults to portrait `1024x1536` |
| `IMAGE_QUALITY` | Image quality (`low`, `medium`, `high`, `xhigh`, `max`, or `auto`) |
| `TOP_K` | Maximum number of matching chunks sent with a question |
| `CHUNK_SIZE` | Recursive splitter target length in characters; defaults to `1000` |
| `CHUNK_OVERLAP` | Repeated characters between adjacent chunks; defaults to `150` and must be smaller than `CHUNK_SIZE` |

## Metadata and filtering

Exchange-provided company, ticker/code, exchange, original title/category, publication date, and source URL are stored when present. Financial year, quarter, reporting period, and page/slide are not fabricated when missing. A single identical document returned by both exchanges can be indexed once while retaining both source references.

The sidebar lists indexed companies, years, quarters, categories, and exchanges. You can select multiple years for comparison or mention known company names/tickers and `FY####` periods in a question. Existing Chroma records remain intact; exchange data uses the same collection and does not trigger folder reconciliation.

## Troubleshooting

- **`python` is not recognized:** Reinstall Python with the PATH option enabled, open a new terminal, or use `py -3.12` when creating the environment.
- **Package installation fails:** Confirm Python is 3.11 or 3.12 and that the virtual environment is active. Upgrade pip and retry the install command. Sentence-transformers/PyTorch downloads are large; check disk space and network access.
- **PowerShell will not activate `.venv`:** Use Command Prompt activation, or set the execution policy to `Bypass` for the current PowerShell process only as shown above.
- **OpenAI key missing or rejected:** Check the `OPENAI_API_KEY` value in `.env`, ensure the file is beside the app, and restart Streamlit. Confirm the key is active and API billing/quota is available. Never put the key in Python code.
- **Image generation not available:** Confirm the `IMAGE_MODEL` name is currently supported and enabled for your OpenAI organization/project. Check organization verification, billing, and API quota. Text access to `gpt-4o-mini` does not guarantee image-model access.
- **Image size rejected:** Use one of the application's supported values: `1024x1536`, `1024x1024`, or `1536x1024`. These are API image sizes; arbitrary A4 pixel dimensions are not assumed.
- **Infographic omits a fact:** Only quotations found exactly in retrieved chunks are rendered as factual labels. Try a more specific request or verify that the relevant presentation was indexed and the requested information appears in its text.
- **Embedding model download fails:** Check internet access, free disk space, and Hugging Face availability. The model downloads the first time it is loaded and is cached for later runs.
- **Embedding dimension error:** Keep `EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2`. This application requires its 768-dimensional output.
- **No filings found:** Confirm the exact exchange symbol/scrip code, date range, company name, and selected categories. Use the official exchange search link if the exchange endpoint is unavailable or has limited history.
- **BSE returns HTTP 403:** This environment's direct BSE endpoint request was denied. The app will not retry around that restriction; open the official BSE announcements search page instead. NSE search can still be used independently.
- **PDF or PowerPoint extraction error:** Confirm the document opens normally and is not damaged or password-protected. Scanned/image-only PDFs need OCR before this application can retrieve their contents.
- **ChromaDB permission or database error:** Close other app instances using the same database, confirm the project folder is writable, and check that `CHROMA_DB_PATH` points to a local writable directory. Do not delete `chroma_db` unless you intend to rebuild the index.
- **The answer says filing evidence is missing:** Search and index the relevant filing, narrow the retrieval filters, and try a specific question. Market questions use Yahoo Finance; its values are latest available, not guaranteed real-time.

## Privacy and cost

PDF/PPTX extraction, embeddings, Pillow composition, the filing cache, and ChromaDB index run locally. The embedding model is downloaded from Hugging Face on first use. The question is sent to OpenAI for intent classification; the financial-analysis request sends GPT-4o-mini the selected NSE/BSE evidence and/or structured Yahoo Finance data, along with relevant conversation history. For image requests, the question and filing excerpts are sent to GPT-4o-mini to prepare the brief, and that brief is sent to the image API. Yahoo Finance requests use the unofficial yfinance interface. The API key stays in `.env` and is not written to ChromaDB.

OpenAI usage may incur API charges. Costs depend on the length of questions, retrieved excerpts, conversation history, model pricing, image size/quality, and how often you ask for an image. Local document indexing does not call the OpenAI API. Check your OpenAI account's current pricing and usage limits.

This tool is for financial research and is not financial advice. Yahoo Finance data may be delayed or incomplete; verify important figures from appropriate primary sources.