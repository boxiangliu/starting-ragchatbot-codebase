"""Tests for AIGenerator and tool calling"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from ai_generator import AIGenerator


class TestAIGeneratorInitialization:
    """Test AIGenerator initialization"""

    def test_initialization(self, ai_generator):
        """Test that AIGenerator initializes correctly"""
        assert ai_generator is not None
        assert ai_generator.model is not None
        assert ai_generator.client is not None
        assert hasattr(ai_generator, 'SYSTEM_PROMPT')
        assert 'search_course_content' in ai_generator.SYSTEM_PROMPT
        assert 'get_course_outline' in ai_generator.SYSTEM_PROMPT


class TestAIGeneratorSystemPrompt:
    """Test system prompt configuration"""

    def test_system_prompt_includes_tools(self):
        """Test that system prompt mentions both tools"""
        prompt = AIGenerator.SYSTEM_PROMPT

        assert 'search_course_content' in prompt
        assert 'get_course_outline' in prompt
        assert 'One tool use per query maximum' in prompt

    def test_system_prompt_includes_outline_instructions(self):
        """Test that system prompt has instructions for course outlines"""
        prompt = AIGenerator.SYSTEM_PROMPT

        assert 'course outline' in prompt.lower() or 'course structure' in prompt.lower()
        assert 'lesson list' in prompt.lower() or 'complete list' in prompt.lower()


class TestAIGeneratorToolCalling:
    """Test tool calling functionality"""

    @pytest.mark.skip(reason="Skipping API tests by default - requires valid API key")
    def test_generate_response_with_tools(self, ai_generator, tool_manager):
        """Test that generate_response can use tools"""
        query = "What is machine learning?"
        tools = tool_manager.get_tool_definitions()

        print(f"\n=== Test: Generate Response with Tools ===")
        print(f"Query: {query}")
        print(f"Tools available: {[t['name'] for t in tools]}")

        try:
            response = ai_generator.generate_response(
                query=query,
                tools=tools,
                tool_manager=tool_manager
            )

            print(f"Response: {response}")
            print(f"Response type: {type(response)}")

            assert isinstance(response, str)
            assert len(response) > 0

        except Exception as e:
            print(f"Error: {e}")
            print(f"Error type: {type(e)}")
            # If API key is missing, that's expected
            if "api" in str(e).lower() or "key" in str(e).lower():
                pytest.skip("API key not configured")
            raise

    def test_handle_tool_execution_structure(self, ai_generator):
        """Test the structure of _handle_tool_execution method"""
        assert hasattr(ai_generator, '_handle_tool_execution')
        import inspect
        sig = inspect.signature(ai_generator._handle_tool_execution)
        params = list(sig.parameters.keys())

        print(f"\n=== Test: Tool Execution Method Structure ===")
        print(f"Parameters: {params}")

        assert 'initial_response' in params
        assert 'base_params' in params
        assert 'tool_manager' in params


class TestAIGeneratorMocked:
    """Test AIGenerator with mocked Anthropic API"""

    def test_tool_use_response_handling(self, test_config):
        """Test handling of tool_use responses"""
        # Create mock client
        mock_client = MagicMock()

        # Mock the initial response with tool use
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = MagicMock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]

        # Mock the final response after tool execution
        mock_final_response = MagicMock()
        mock_final_content = MagicMock()
        mock_final_content.text = "This is the final answer"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        # Create AI generator with mocked client
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        generator.client = mock_client

        # Create mock tool manager
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Test
        print(f"\n=== Test: Mocked Tool Use Response ===")

        result = generator.generate_response(
            query="test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        print(f"Result: {result}")
        print(f"Tool manager execute_tool called: {mock_tool_manager.execute_tool.called}")
        print(f"Number of API calls: {mock_client.messages.create.call_count}")

        assert result == "This is the final answer"
        assert mock_tool_manager.execute_tool.called
        assert mock_client.messages.create.call_count == 2

    def test_direct_response_handling(self, test_config):
        """Test handling of direct responses (no tool use)"""
        # Create mock client
        mock_client = MagicMock()

        # Mock a direct text response
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_content = MagicMock()
        mock_content.text = "Direct answer without tools"
        mock_response.content = [mock_content]

        mock_client.messages.create.return_value = mock_response

        # Create AI generator with mocked client
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        generator.client = mock_client

        # Test
        print(f"\n=== Test: Mocked Direct Response ===")

        result = generator.generate_response(
            query="What is 2+2?",
            tools=[],
            tool_manager=None
        )

        print(f"Result: {result}")
        print(f"Number of API calls: {mock_client.messages.create.call_count}")

        assert result == "Direct answer without tools"
        assert mock_client.messages.create.call_count == 1

    def test_tool_execution_error_handling(self, test_config):
        """Test handling when tool execution fails"""
        # Create mock client
        mock_client = MagicMock()

        # Mock tool use response
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = MagicMock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_456"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]

        # Mock final response
        mock_final_response = MagicMock()
        mock_final_content = MagicMock()
        mock_final_content.text = "Error handled response"
        mock_final_response.content = [mock_final_content]

        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]

        # Create AI generator
        generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
        generator.client = mock_client

        # Mock tool manager that returns error
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool.return_value = "Tool error: Something went wrong"

        print(f"\n=== Test: Tool Execution Error ===")

        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        print(f"Result: {result}")
        print(f"Tool execution result: {mock_tool_manager.execute_tool.return_value}")

        # Should still get a response even if tool fails
        assert isinstance(result, str)
        assert len(result) > 0
