# Spec-to-Sprint

An AI tool to parse engineering specifications and generate Notion tasks.

## Description
Spec-to-Sprint is a Python-based AI project that processes engineering specification documents and automatically generates task lists in Notion. It uses a combination of PDF processing, RAG (Retrieval-Augmented Generation), and LLM (Large Language Model) techniques to extract and organize tasks from technical documents.

## Setup Instructions

1. Install Python 3.11
2. Create and activate virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

## Running the Application

1. Start the backend:
   ```bash
   python backend/main.py
   ```

2. Serve the frontend:
   ```bash
   python -m http.server 8000
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000/frontend/
   ```

## Project Structure

```
spec-to-sprint/
├── backend/
│   ├── main.py              # Main script
│   ├── requirements.txt     # Python dependencies
│   ├── pdf_processor.py     # PDF text extraction
│   ├── rag_pipeline.py      # RAG implementation
│   ├── llm_tasks.py         # LLM task extraction
│   ├── notion_integration.py # Notion API integration
├── frontend/
│   ├── index.html           # Main HTML file
│   ├── script.js            # Frontend JavaScript
│   ├── styles.css           # Tailwind CSS styles
├── data/
│   ├── specs/               # Sample PDF specs
│   ├── annotations/         # Annotated task CSVs
```

## Development Notes

- The project uses free tools and is optimized for local development
- All dependencies are listed in backend/requirements.txt
- The frontend is compatible with Lovable.dev for export
- Python files are formatted with Black and linted with Flake8