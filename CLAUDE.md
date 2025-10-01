# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) system that enables semantic search and AI-powered Q&A over course documents. The system uses an **agentic approach** where Claude autonomously decides when to search course materials versus using existing knowledge.

## Running the Application

**Quick start:**
```bash
./run.sh
```

**Manual start:**
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web interface: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

**Package management:**
- Uses `uv` (not pip) for all dependency management
- Install dependencies: `uv sync`
- Add new package: `uv add <package>`

## Environment Setup

Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Architecture

### Core Design Pattern: Agentic RAG with Tool Calling

The system implements a **two-phase Claude interaction** pattern:

1. **Phase 1**: Claude receives user query + tool definitions → decides whether to search or answer directly
2. **Phase 2** (if tool used): Claude receives tool results → synthesizes final answer

**Key architectural decision**: Claude has access to `search_course_content` tool but is instructed to use "**one search per query maximum**" via system prompt in `ai_generator.py`.

### Component Flow

```
User Query → FastAPI → RAGSystem → AIGenerator → Claude API (Phase 1)
                                         ↓
                                    Tool Decision?
                                         ↓
                           Yes: ToolManager → CourseSearchTool → VectorStore → ChromaDB
                                         ↓
                                    Claude API (Phase 2) → Final Answer
```

### Two-Collection Strategy

The system uses **two ChromaDB collections** with distinct purposes:

1. **`course_catalog`**: Stores course metadata (titles, instructors, lesson lists)
   - Purpose: Fuzzy course name resolution via semantic search
   - Example: "intro" → "Introduction to RAG"

2. **`course_content`**: Stores chunked course material with embeddings
   - Purpose: Filtered semantic search within resolved courses/lessons
   - Metadata: course_title, lesson_number, chunk_index

This separation enables efficient course resolution before content search.

### Document Processing Pipeline

Files in `docs/` are processed on startup (`app.py:startup_event`):

1. **Parse** (`document_processor.py`): Extract course metadata and lessons from structured text format
   ```
   Course Title: [title]
   Course Link: [url]
   Course Instructor: [name]

   Lesson 0: [title]
   Lesson Link: [url]
   [content...]
   ```

2. **Chunk** (`document_processor.py:chunk_text`): Sentence-aware splitting with configurable overlap
   - Default: 800 chars per chunk, 100 char overlap
   - Handles abbreviations (Dr., U.S., etc.)
   - **Context injection**: Prepends "Course [X] Lesson [Y] content:" to chunks

3. **Store** (`vector_store.py`): Add to both collections with sentence-transformers embeddings

### Session Management

Conversation history is tracked per session (default: 2 exchanges = 4 messages). History is passed to Claude in the system context to enable coherent multi-turn conversations like "Can you explain more about that?"

## Key Files

- **`rag_system.py`**: Main orchestrator coordinating all components
- **`ai_generator.py`**: Claude API interface with tool execution loop
- **`vector_store.py`**: ChromaDB manager with two-collection logic
- **`search_tools.py`**: Tool definitions and ToolManager for Claude function calling
- **`document_processor.py`**: Document parsing and sentence-aware chunking
- **`app.py`**: FastAPI endpoints and startup document loading
- **`config.py`**: Configuration (chunk size, model names, API keys)

## Configuration (`backend/config.py`)

Important settings:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (sentence-transformers)
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 chunks per search
- `MAX_HISTORY`: 2 conversation exchanges
- `CHROMA_PATH`: "./chroma_db"

## Adding New Course Documents

Place `.txt`, `.pdf`, or `.docx` files in `docs/` folder. They will be automatically loaded on next server restart. The system performs **deduplication by course title** to avoid re-processing existing courses.

Expected file format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [title]
Lesson Link: [url]
[lesson content]

Lesson 1: [title]
...
```

## Tool System Extension

To add new tools for Claude:

1. Create a class implementing `Tool` interface in `search_tools.py`:
   ```python
   class MyTool(Tool):
       def get_tool_definition(self) -> Dict[str, Any]:
           return {
               "name": "my_tool",
               "description": "...",
               "input_schema": {...}
           }

       def execute(self, **kwargs) -> str:
           # Tool logic
           return result
   ```

2. Register in `rag_system.py:__init__`:
   ```python
   my_tool = MyTool()
   self.tool_manager.register_tool(my_tool)
   ```

Claude will automatically have access to the new tool in its tool definitions.

## Important Behavioral Notes

### System Prompt Constraints
The system prompt in `ai_generator.py` enforces:
- "One search per query maximum" - prevents excessive tool use
- Direct answers for general knowledge questions
- Search only for course-specific content
- No meta-commentary about searching or results

### Context Injection Strategy
Chunks are stored with prefixed context (e.g., "Course X Lesson Y content: ...") to improve retrieval quality. This metadata becomes part of the searchable embedding, helping the vector store return more relevant results.

### Inconsistency to Be Aware Of
In `document_processor.py`, there's an inconsistency in context prefixes:
- Most chunks: `"Lesson {N} content: {chunk}"` (line 186)
- Last lesson chunks: `"Course {title} Lesson {N} content: {chunk}"` (line 234)

This doesn't break functionality but creates non-uniform chunk formats.

## API Endpoints

**POST /api/query**
```json
Request: {"query": "What is RAG?", "session_id": "session_1"}
Response: {"answer": "...", "sources": ["Course - Lesson 2"], "session_id": "session_1"}
```

**GET /api/courses**
```json
Response: {"total_courses": 4, "course_titles": ["Introduction to RAG", ...]}
```

## ChromaDB Persistence

ChromaDB data persists in `backend/chroma_db/` directory. To reset the database:
1. Stop the server
2. Delete `backend/chroma_db/`
3. Restart server - documents will be reprocessed from `docs/`