"""Integration tests for RAGSystem"""
import pytest
import os
import shutil
from unittest.mock import MagicMock, patch
from rag_system import RAGSystem
from models import Course, Lesson


class TestRAGSystemInitialization:
    """Test RAG system initialization"""

    def test_rag_system_components(self, test_config):
        """Test that RAG system initializes all components"""
        # Clean up test DB
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)

        print(f"\n=== Test: RAG System Components ===")
        print(f"Has document_processor: {hasattr(system, 'document_processor')}")
        print(f"Has vector_store: {hasattr(system, 'vector_store')}")
        print(f"Has ai_generator: {hasattr(system, 'ai_generator')}")
        print(f"Has session_manager: {hasattr(system, 'session_manager')}")
        print(f"Has tool_manager: {hasattr(system, 'tool_manager')}")
        print(f"Has search_tool: {hasattr(system, 'search_tool')}")
        print(f"Has outline_tool: {hasattr(system, 'outline_tool')}")

        assert hasattr(system, 'document_processor')
        assert hasattr(system, 'vector_store')
        assert hasattr(system, 'ai_generator')
        assert hasattr(system, 'session_manager')
        assert hasattr(system, 'tool_manager')
        assert hasattr(system, 'search_tool')
        assert hasattr(system, 'outline_tool')

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

    def test_tools_registered(self, test_config):
        """Test that tools are properly registered"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)
        tools = system.tool_manager.get_tool_definitions()

        print(f"\n=== Test: Tools Registered ===")
        print(f"Number of tools: {len(tools)}")
        print(f"Tool names: {[t['name'] for t in tools]}")

        assert len(tools) == 2
        tool_names = [t['name'] for t in tools]
        assert 'search_course_content' in tool_names
        assert 'get_course_outline' in tool_names

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)


class TestRAGSystemQuery:
    """Test RAG system query functionality"""

    def test_query_method_exists(self, test_config):
        """Test that query method exists and has correct signature"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)

        print(f"\n=== Test: Query Method ===")
        print(f"Has query method: {hasattr(system, 'query')}")

        assert hasattr(system, 'query')

        import inspect
        sig = inspect.signature(system.query)
        params = list(sig.parameters.keys())

        print(f"Query parameters: {params}")

        assert 'query' in params
        assert 'session_id' in params

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

    def test_query_with_mocked_ai(self, test_config, sample_course, sample_chunks):
        """Test query with mocked AI response"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)

        # Add test data
        system.vector_store.add_course_metadata(sample_course)
        system.vector_store.add_course_content(sample_chunks)

        # Mock the AI generator
        mock_response = "Machine learning is a subset of AI that enables systems to learn from data."
        system.ai_generator.generate_response = MagicMock(return_value=mock_response)

        print(f"\n=== Test: Query with Mocked AI ===")

        response, sources = system.query("What is machine learning?")

        print(f"Response: {response}")
        print(f"Sources: {sources}")
        print(f"AI generator called: {system.ai_generator.generate_response.called}")

        assert response == mock_response
        assert isinstance(sources, list)
        assert system.ai_generator.generate_response.called

        # Check that tools were passed to AI
        call_args = system.ai_generator.generate_response.call_args
        assert call_args is not None
        assert 'tools' in call_args.kwargs
        assert 'tool_manager' in call_args.kwargs

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

    def test_query_passes_tools_to_ai(self, test_config):
        """Test that query passes tool definitions to AI generator"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)

        # Mock the AI generator to inspect what it receives
        system.ai_generator.generate_response = MagicMock(return_value="test response")

        print(f"\n=== Test: Tools Passed to AI ===")

        system.query("test query")

        # Check the call
        call_args = system.ai_generator.generate_response.call_args

        print(f"Call args: {call_args}")
        print(f"Has 'tools' kwarg: {'tools' in call_args.kwargs}")
        print(f"Has 'tool_manager' kwarg: {'tool_manager' in call_args.kwargs}")

        if 'tools' in call_args.kwargs:
            tools = call_args.kwargs['tools']
            print(f"Tools passed: {[t['name'] for t in tools] if tools else None}")

        assert call_args is not None
        assert 'tools' in call_args.kwargs
        assert 'tool_manager' in call_args.kwargs

        tools = call_args.kwargs['tools']
        assert len(tools) == 2
        assert any(t['name'] == 'search_course_content' for t in tools)

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)


class TestRAGSystemToolExecution:
    """Test that RAG system correctly executes tools during queries"""

    def test_search_tool_accessible_from_rag_system(self, test_config, sample_course, sample_chunks):
        """Test that search tool can be executed through RAG system's tool manager"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)

        # Add test data
        system.vector_store.add_course_metadata(sample_course)
        system.vector_store.add_course_content(sample_chunks)

        print(f"\n=== Test: Search Tool Execution ===")

        # Try to execute search tool directly
        result = system.tool_manager.execute_tool(
            "search_course_content",
            query="machine learning"
        )

        print(f"Tool execution result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be an error message
        assert "not found" not in result.lower() or "No relevant content" in result

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

    def test_outline_tool_accessible_from_rag_system(self, test_config, sample_course, sample_chunks):
        """Test that outline tool can be executed through RAG system's tool manager"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)

        # Add test data
        system.vector_store.add_course_metadata(sample_course)
        system.vector_store.add_course_content(sample_chunks)

        print(f"\n=== Test: Outline Tool Execution ===")

        # Try to execute outline tool directly
        result = system.tool_manager.execute_tool(
            "get_course_outline",
            course_name="Test Course"
        )

        print(f"Tool execution result: {result}")

        assert isinstance(result, str)
        assert len(result) > 0
        # Should include course info
        assert "Course:" in result or "No course found" in result

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)


class TestRAGSystemWithData:
    """Test RAG system behavior with actual course data"""

    @pytest.fixture
    def rag_system_with_data(self, test_config, sample_course, sample_chunks):
        """Create RAG system with test data"""
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

        system = RAGSystem(test_config)
        system.vector_store.add_course_metadata(sample_course)
        system.vector_store.add_course_content(sample_chunks)

        yield system

        # Cleanup
        if os.path.exists(test_config.CHROMA_PATH):
            shutil.rmtree(test_config.CHROMA_PATH)

    def test_content_search_through_tool_manager(self, rag_system_with_data):
        """Test content search executed through tool manager"""
        result = rag_system_with_data.tool_manager.execute_tool(
            "search_course_content",
            query="neural networks"
        )

        print(f"\n=== Test: Content Search via Tool Manager ===")
        print(f"Result: {result}")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_vector_store_directly(self, rag_system_with_data):
        """Test vector store search directly"""
        result = rag_system_with_data.vector_store.search(
            query="neural networks"
        )

        print(f"\n=== Test: Direct Vector Store Search ===")
        print(f"Result type: {type(result)}")
        print(f"Has error: {result.error}")
        print(f"Is empty: {result.is_empty()}")
        print(f"Num documents: {len(result.documents)}")
        if result.documents:
            print(f"First document: {result.documents[0][:100]}...")

        assert result is not None
        assert result.error is None

    def test_search_tool_directly(self, rag_system_with_data):
        """Test search tool execute method directly"""
        result = rag_system_with_data.search_tool.execute(
            query="machine learning"
        )

        print(f"\n=== Test: Direct Search Tool Execute ===")
        print(f"Result: {result}")
        print(f"Type: {type(result)}")

        assert isinstance(result, str)
        assert len(result) > 0
