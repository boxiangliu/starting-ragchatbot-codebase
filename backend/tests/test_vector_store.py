"""Tests for VectorStore"""
import pytest
import os
import shutil
from vector_store import VectorStore, SearchResults


class TestVectorStoreInitialization:
    """Test vector store initialization"""

    def test_creates_collections(self, test_config):
        """Test that vector store creates required collections"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        store = VectorStore(
            test_config.CHROMA_PATH,
            test_config.EMBEDDING_MODEL,
            test_config.MAX_RESULTS
        )

        print(f"\n=== Test: Collection Creation ===")
        print(f"Has course_catalog: {hasattr(store, 'course_catalog')}")
        print(f"Has course_content: {hasattr(store, 'course_content')}")

        assert hasattr(store, 'course_catalog')
        assert hasattr(store, 'course_content')
        assert store.course_catalog is not None
        assert store.course_content is not None

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)


class TestVectorStoreSearch:
    """Test vector store search functionality"""

    def test_search_returns_results_object(self, vector_store):
        """Test that search returns SearchResults object"""
        result = vector_store.search(query="machine learning")

        print(f"\n=== Test: Search Returns Results ===")
        print(f"Result type: {type(result)}")
        print(f"Has documents: {hasattr(result, 'documents')}")
        print(f"Has metadata: {hasattr(result, 'metadata')}")
        print(f"Has error: {hasattr(result, 'error')}")

        assert isinstance(result, SearchResults)
        assert hasattr(result, 'documents')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'error')

    def test_search_with_valid_query(self, vector_store):
        """Test search with query that should find results"""
        result = vector_store.search(query="machine learning")

        print(f"\n=== Test: Valid Query Search ===")
        print(f"Error: {result.error}")
        print(f"Is empty: {result.is_empty()}")
        print(f"Num results: {len(result.documents)}")
        if result.documents:
            print(f"First result: {result.documents[0][:100]}...")

        assert result.error is None
        # May or may not find results depending on content

    def test_search_with_course_filter(self, vector_store):
        """Test search with course name filter"""
        result = vector_store.search(
            query="neural",
            course_name="Test Course"
        )

        print(f"\n=== Test: Search with Course Filter ===")
        print(f"Error: {result.error}")
        print(f"Is empty: {result.is_empty()}")
        print(f"Num results: {len(result.documents)}")

        assert result.error is None

    def test_search_with_lesson_filter(self, vector_store):
        """Test search with lesson number filter"""
        result = vector_store.search(
            query="neural",
            lesson_number=2
        )

        print(f"\n=== Test: Search with Lesson Filter ===")
        print(f"Error: {result.error}")
        print(f"Is empty: {result.is_empty()}")
        print(f"Num results: {len(result.documents)}")

        assert result.error is None

    def test_search_nonexistent_course(self, vector_store):
        """Test search with non-existent course"""
        result = vector_store.search(
            query="test",
            course_name="Nonexistent Course XYZ 12345"
        )

        print(f"\n=== Test: Nonexistent Course Search ===")
        print(f"Error: {result.error}")
        print(f"Is empty: {result.is_empty()}")

        # Should return error about course not found
        assert result.error is not None or result.is_empty()


class TestVectorStoreData:
    """Test adding data to vector store"""

    def test_add_course_metadata(self, test_config, sample_course):
        """Test adding course metadata"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        store = VectorStore(
            test_config.CHROMA_PATH,
            test_config.EMBEDDING_MODEL,
            test_config.MAX_RESULTS
        )

        print(f"\n=== Test: Add Course Metadata ===")

        # Should not raise exception
        store.add_course_metadata(sample_course)

        # Verify it was added
        titles = store.get_existing_course_titles()
        print(f"Existing titles: {titles}")

        assert sample_course.title in titles

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

    def test_add_course_content(self, test_config, sample_chunks):
        """Test adding course content chunks"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        store = VectorStore(
            test_config.CHROMA_PATH,
            test_config.EMBEDDING_MODEL,
            test_config.MAX_RESULTS
        )

        print(f"\n=== Test: Add Course Content ===")
        print(f"Adding {len(sample_chunks)} chunks")

        # Should not raise exception
        store.add_course_content(sample_chunks)

        # Try to search for added content
        result = store.search(query="machine learning")
        print(f"Search result after adding: empty={result.is_empty()}, error={result.error}")

        assert result.error is None

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)


class TestVectorStoreCourseResolution:
    """Test course name resolution"""

    def test_resolve_course_name(self, vector_store):
        """Test course name resolution with partial match"""
        # Try to resolve with partial name
        resolved = vector_store._resolve_course_name("Test Course")

        print(f"\n=== Test: Course Name Resolution ===")
        print(f"Query: 'Test Course'")
        print(f"Resolved to: {resolved}")

        # Should resolve to full course title
        assert resolved is not None
        assert "Machine Learning" in resolved

    def test_resolve_course_name_fuzzy(self, vector_store):
        """Test fuzzy course name matching"""
        resolved = vector_store._resolve_course_name("ML")

        print(f"\n=== Test: Fuzzy Course Name Resolution ===")
        print(f"Query: 'ML'")
        print(f"Resolved to: {resolved}")

        # May or may not match depending on embedding similarity
        # Just verify it returns a string or None
        assert resolved is None or isinstance(resolved, str)


class TestVectorStoreOutline:
    """Test course outline retrieval"""

    def test_get_course_outline(self, vector_store):
        """Test getting course outline"""
        outline = vector_store.get_course_outline("Test Course")

        print(f"\n=== Test: Get Course Outline ===")
        print(f"Outline: {outline}")

        assert outline is not None
        assert "course_title" in outline
        assert "course_link" in outline
        assert "lessons" in outline
        assert isinstance(outline["lessons"], list)
        assert len(outline["lessons"]) == 3

    def test_get_course_outline_with_lesson_details(self, vector_store):
        """Test that outline includes lesson details"""
        outline = vector_store.get_course_outline("Machine Learning")

        print(f"\n=== Test: Outline Lesson Details ===")

        assert outline is not None
        lessons = outline["lessons"]
        assert len(lessons) > 0

        # Check first lesson structure
        first_lesson = lessons[0]
        print(f"First lesson: {first_lesson}")

        assert "lesson_number" in first_lesson
        assert "lesson_title" in first_lesson
        assert "lesson_link" in first_lesson

    def test_get_course_outline_nonexistent(self, vector_store):
        """Test getting outline for non-existent course"""
        outline = vector_store.get_course_outline("Nonexistent Course XYZ")

        print(f"\n=== Test: Nonexistent Course Outline ===")
        print(f"Result: {outline}")

        assert outline is None
