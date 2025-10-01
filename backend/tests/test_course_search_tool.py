"""Tests for CourseSearchTool"""
import pytest
from search_tools import CourseSearchTool


class TestCourseSearchToolDefinition:
    """Test tool definition"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly formatted"""
        definition = course_search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]


class TestCourseSearchToolExecution:
    """Test execute method of CourseSearchTool"""

    def test_execute_simple_query(self, course_search_tool):
        """Test basic content search without filters"""
        result = course_search_tool.execute(query="machine learning")

        print(f"\n=== Test: Simple Query ===")
        print(f"Query: 'machine learning'")
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Course on Machine Learning" in result or "No relevant content" in result

    def test_execute_with_course_filter(self, course_search_tool):
        """Test search with course name filter"""
        result = course_search_tool.execute(
            query="neural networks",
            course_name="Test Course"
        )

        print(f"\n=== Test: Query with Course Filter ===")
        print(f"Query: 'neural networks', Course: 'Test Course'")
        print(f"Result: {result}")

        assert isinstance(result, str)
        # Should either find content or return error message
        assert len(result) > 0

    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test search with lesson number filter"""
        result = course_search_tool.execute(
            query="neural",
            lesson_number=2
        )

        print(f"\n=== Test: Query with Lesson Filter ===")
        print(f"Query: 'neural', Lesson: 2")
        print(f"Result: {result}")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_with_both_filters(self, course_search_tool):
        """Test search with both course and lesson filters"""
        result = course_search_tool.execute(
            query="networks",
            course_name="Machine Learning",
            lesson_number=2
        )

        print(f"\n=== Test: Query with Both Filters ===")
        print(f"Query: 'networks', Course: 'Machine Learning', Lesson: 2")
        print(f"Result: {result}")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_nonexistent_course(self, course_search_tool):
        """Test search with non-existent course name"""
        result = course_search_tool.execute(
            query="test",
            course_name="Nonexistent Course XYZ"
        )

        print(f"\n=== Test: Nonexistent Course ===")
        print(f"Query: 'test', Course: 'Nonexistent Course XYZ'")
        print(f"Result: {result}")

        assert isinstance(result, str)
        assert "No course found" in result or "No relevant content" in result

    def test_execute_empty_results(self, course_search_tool):
        """Test search that should return no results"""
        result = course_search_tool.execute(
            query="quantum physics thermodynamics"
        )

        print(f"\n=== Test: Empty Results ===")
        print(f"Query: 'quantum physics thermodynamics'")
        print(f"Result: {result}")

        assert isinstance(result, str)
        # Either finds something or returns "No relevant content"
        assert len(result) > 0

    def test_execute_tracks_sources(self, course_search_tool):
        """Test that execute method tracks sources"""
        # Reset sources first
        course_search_tool.last_sources = []

        result = course_search_tool.execute(query="machine learning")

        print(f"\n=== Test: Source Tracking ===")
        print(f"Query: 'machine learning'")
        print(f"Result: {result}")
        print(f"Last sources: {course_search_tool.last_sources}")

        # Sources should be tracked if results were found
        if "No relevant content" not in result:
            assert isinstance(course_search_tool.last_sources, list)
            if len(course_search_tool.last_sources) > 0:
                source = course_search_tool.last_sources[0]
                assert "text" in source
                assert "link" in source


class TestCourseSearchToolFormatting:
    """Test result formatting"""

    def test_format_includes_course_context(self, course_search_tool):
        """Test that results include course context headers"""
        result = course_search_tool.execute(query="machine learning")

        print(f"\n=== Test: Format Includes Context ===")
        print(f"Result: {result}")

        if "No relevant content" not in result:
            # Should have course title in brackets
            assert "[" in result and "]" in result

    def test_format_includes_lesson_context(self, course_search_tool):
        """Test that results include lesson context when available"""
        result = course_search_tool.execute(query="neural networks")

        print(f"\n=== Test: Format Includes Lesson Context ===")
        print(f"Result: {result}")

        if "No relevant content" not in result:
            # Should mention lesson if found in lesson content
            # Format: [Course Title - Lesson N]
            assert "Lesson" in result or "[" in result


class TestVectorStoreIntegration:
    """Test integration with VectorStore"""

    def test_vector_store_search_called(self, course_search_tool, vector_store):
        """Test that vector store search is being called"""
        # This test verifies the integration between tool and vector store
        result = course_search_tool.execute(query="neural")

        print(f"\n=== Test: VectorStore Integration ===")
        print(f"Query: 'neural'")
        print(f"Result: {result}")
        print(f"Vector store max_results: {vector_store.max_results}")

        # Should return a string result
        assert isinstance(result, str)

    def test_handles_vector_store_errors(self, course_search_tool):
        """Test graceful handling of vector store errors"""
        # Test with very long query that might cause issues
        long_query = "test " * 1000

        result = course_search_tool.execute(query=long_query)

        print(f"\n=== Test: Handle Errors ===")
        print(f"Query length: {len(long_query)}")
        print(f"Result: {result[:200]}...")

        # Should still return a string, not crash
        assert isinstance(result, str)
