# Stock Market Analysis RAG Chatbot

A local Streamlit application for asking questions about investor presentations in PDF and PowerPoint format. It extracts document text, creates local embeddings with `sentence-transformers/all-mpnet-base-v2`, stores searchable chunks in persistent ChromaDB, and sends the question plus matching excerpts to OpenAI `gpt-4o-mini` for an answer. Explicit image or infographic requests first use the same ChromaDB retrieval, then use GPT-4o-mini to prepare a grounded brief and the OpenAI Images API to create an image.

The chatbot answers only from the presentations you index. It does **not** retrieve live stock prices, news, filings, or other market data. It is not investment advice.

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

### 2. Create the project and presentations folders

Open PowerShell and run:

```powershell
New-Item -ItemType Directory -Force "$HOME\Desktop\Equity Lens"
New-Item -ItemType Directory -Force "$HOME\Desktop\investor_presentations"
```

Keep `stock_rag_chatbot.py`, `.env`, `.gitignore`, `requirements.txt`, and `README.md` in the project folder. Place your `.pdf` and `.pptx` presentations directly inside `investor_presentations`.

### 3. Configure `.env`

Copy `.env.example` to `.env` in the project folder. Replace `YOUR_USERNAME` in `DOCUMENTS_FOLDER` with your Windows account folder name, or set it to another folder containing your presentations. Replace the API key placeholder with your own key:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
DOCUMENTS_FOLDER=C:/Users/YOUR_USERNAME/Desktop/investor_presentations
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

Do not add quotes around ordinary values. Keep the key private: `.env` is excluded by `.gitignore`; never commit it, paste it into source code, or share it. The chatbot displays a folder override in its sidebar for the current run; edit `.env` to make the location permanent.

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

This command starts Streamlit's Uvicorn server and opens the local app in your browser (normally at `http://localhost:8501`). Keep the terminal open while using the app; press `Ctrl+C` there to stop the server. Add presentations to the configured folder, then choose **Index / Refresh Documents** in the sidebar. Wait for indexing to finish and ask a question in the chat box. The answer includes expandable source filenames, page or slide locations, and retrieved text excerpts.

## RAG pipeline

1. The app reads `.pdf` and `.pptx` files directly inside `DOCUMENTS_FOLDER`. PDFs are extracted page by page; PowerPoint files are extracted slide by slide, including table text. Empty pages and slides are skipped.
2. Before indexing, review each file's metadata in **Review document metadata**. A standardized filename can prefill the form: `Company__Ticker__FY2026__Q1__Investor_Presentation.pdf`. If a filename does not match that format, enter company, ticker, financial year, quarter, and document type yourself. The app does not guess missing values.
3. Each page or slide is split independently with `RecursiveCharacterTextSplitter`. It prefers paragraph breaks, then line and sentence boundaries, then word and character boundaries. `CHUNK_SIZE` (default `1000`) and `CHUNK_OVERLAP` (default `150`) are character counts set in `.env`; every chunk keeps its original page or slide label.
4. The app embeds each chunk with Hugging Face `sentence-transformers/all-mpnet-base-v2`. The app checks that the model returns **768-dimensional vectors**. Question vectors are generated with that same model and normalization setting.
5. One persistent ChromaDB collection stores every company's chunks. Each chunk has `company`, `ticker`, `fiscal_year`, `quarter`, `document_type`, `source`, `source_path`, `file_hash`, `chunk_number`, and `location`, alongside its text and vector.
6. ChromaDB applies metadata `where` filters before semantic ranking. Selected company, year, and quarter fields combine with AND; multiple values use Chroma's `$in` operator. The retrieval function also accepts explicit `where_document` text-contains filters. With no selected or recognized filters, it searches across all indexed companies and years.
7. Multi-year questions are retrieved separately for each requested financial year and then combined with year labels intact. The answer instructions require the model to keep each period separate. A missing company/year/quarter match is reported instead of widening the search.
8. File path and content hashes form unique chunk IDs. Re-indexing uses `upsert`, so repeated runs do not duplicate chunks; changed content receives new IDs and old chunks are removed only after replacement succeeds. Metadata-only edits update the same IDs. **Rebuild selected folder index** safely re-reads that folder without deleting the database or another folder's records. Existing databases remain readable; re-index files after reviewing metadata to migrate old records in place.
9. ChromaDB returns the `TOP_K` semantically similar excerpts within the requested filter. OpenAI receives company, ticker, year, quarter, filename, and page/slide labels and is instructed not to mix values across periods.

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

The tests mock ChromaDB results and OpenAI calls; they do not spend API credits. In the app, index presentations first, then try these prompts:

| Prompt | Expected behavior |
| --- | --- |
| `What was revenue in FY2025?` | Text answer only |
| `Summarize the investor presentation` | Text answer only |
| `Create an A4 infographic showing revenue growth` | Retrieve evidence, generate and display a PNG infographic |
| `Create a graph of revenue growth by fiscal year` | Draw a graph only if at least two comparable, source-verified fiscal-year values are retrieved; otherwise explain which data is missing |
| `Generate an infographic about the company's business segments` | Retrieve evidence, generate and display a PNG infographic |
| `What information would a revenue infographic contain?` | Text answer only |

Also check the missing-key and simulated image-API failure tests in the command above; they verify the app returns useful errors instead of crashing.

Scanned PDFs that contain page images but no selectable text are not OCR'd by this app. Use OCR software to make them searchable before indexing. If a PDF has no extractable text, the app reports it as a file-level indexing error.

## Configuration

| Setting | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Secret key used only for answer generation through OpenAI |
| `DOCUMENTS_FOLDER` | Folder containing investor presentation PDFs and PPTX files |
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

The recommended filename is `Company__Ticker__FY2026__Q1__Investor_Presentation.pdf` (or `.pptx`). Use double underscores between the five fields. The financial year uses `FY2026`; quarter can be `Q1` through `Q4`, `FY`, or `Annual`. Filename extraction only prefills the review form; save checked metadata for every file before indexing. Other filename patterns require manual entry.

The sidebar lists indexed companies, the financial years for the selected company, and available quarters. Leaving a control at “All” does not apply that filter. You can select multiple years for comparison or mention known company names/tickers and `FY####` periods in a question. Source labels include company, ticker, year, quarter, filename, and page/slide. Old chunks without the new fields remain stored until their source files are re-indexed with reviewed metadata; migration never clears the database automatically.

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
- **No documents found:** Check the folder path shown in the sidebar and place PDF/PPTX files directly inside it. The app creates a missing folder, but it cannot find files in the wrong location or nested subfolders.
- **PDF or PowerPoint extraction error:** Confirm the document opens normally and is not damaged or password-protected. Scanned/image-only PDFs need OCR before this application can retrieve their contents.
- **ChromaDB permission or database error:** Close other app instances using the same database, confirm the project folder is writable, and check that `CHROMA_DB_PATH` points to a local writable directory. Do not delete `chroma_db` unless you intend to rebuild the index.
- **The answer says evidence is missing:** Re-index after adding or editing files, and try a more specific question. The app only searches extracted text in indexed presentations; it does not browse the web or retrieve live market data.

## Privacy and cost

PDF/PPTX extraction, embeddings, Pillow composition, and the ChromaDB index run locally. The embedding model is downloaded from Hugging Face on first use. For text answers, the question and retrieved presentation excerpts are sent to OpenAI. For image requests, the question and excerpts are also sent to GPT-4o-mini to prepare the brief, and that brief is sent to the image API. Do not index confidential material unless this use complies with your organization's policies and OpenAI account terms. The API key stays in `.env` and is not written to the ChromaDB records.

OpenAI usage may incur API charges. Costs depend on the length of questions, retrieved excerpts, conversation history, model pricing, image size/quality, and how often you ask for an image. Local document indexing does not call the OpenAI API. Check your OpenAI account's current pricing and usage limits.

This tool is for document analysis only. It does not fetch current prices, market data, or news and should not be used as financial advice.