"""Shared test fixtures for RAG system tests"""
import pytest
import sys
import os
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from config import Config


@pytest.fixture
def test_config():
    """Create test configuration"""
    import tempfile
    config = Config()
    # Use a unique temporary directory for each test run
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    config.CHROMA_PATH = temp_dir
    config.MAX_RESULTS = 3
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course on Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Test",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
            Lesson(lesson_number=1, title="Basics of ML", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/lesson2"),
        ]
    )


@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is an introduction to machine learning. Machine learning is a subset of artificial intelligence.",
            course_title="Test Course on Machine Learning",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes.",
            course_title="Test Course on Machine Learning",
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Deep learning is a subset of machine learning that uses multi-layered neural networks.",
            course_title="Test Course on Machine Learning",
            lesson_number=2,
            chunk_index=2
        ),
    ]


@pytest.fixture
def vector_store(test_config, sample_course, sample_chunks):
    """Create and populate a test vector store"""
    # Clean up any existing test database
    import shutil
    if os.path.exists(test_config.CHROMA_PATH):
        shutil.rmtree(test_config.CHROMA_PATH)

    store = VectorStore(
        test_config.CHROMA_PATH,
        test_config.EMBEDDING_MODEL,
        test_config.MAX_RESULTS
    )

    # Add test data
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_chunks)

    yield store

    # Cleanup after tests
    if os.path.exists(test_config.CHROMA_PATH):
        shutil.rmtree(test_config.CHROMA_PATH)


@pytest.fixture
def course_search_tool(vector_store):
    """Create a CourseSearchTool with test data"""
    return CourseSearchTool(vector_store)


@pytest.fixture
def course_outline_tool(vector_store):
    """Create a CourseOutlineTool with test data"""
    return CourseOutlineTool(vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def ai_generator(test_config):
    """Create an AIGenerator for testing"""
    # Use API key from config if available, otherwise use dummy
    api_key = test_config.ANTHROPIC_API_KEY or "test-api-key"
    return AIGenerator(api_key, test_config.ANTHROPIC_MODEL)
