#!/usr/bin/env python3
"""
LangChain Integration with SageMaker DLC Endpoints

Demonstrates:
- Custom LangChain LLM wrapper for SageMaker
- Conversation chains with memory
- Agents with tools
- Structured output parsing
"""

from langchain.llms.base import LLM
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List
import boto3
import json


class SageMakerLLM(LLM):
    """Custom LangChain wrapper for SageMaker endpoints"""

    endpoint_name: str
    region_name: str = "us-west-2"
    model_kwargs: dict = {}

    @property
    def _llm_type(self) -> str:
        return "sagemaker"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call SageMaker endpoint"""
        runtime = boto3.client('sagemaker-runtime', region_name=self.region_name)

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.model_kwargs.get("max_tokens", 256),
                "temperature": self.model_kwargs.get("temperature", 0.7),
                "top_p": self.model_kwargs.get("top_p", 0.9),
                "stop": stop or [],
            }
        }

        response = runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result = json.loads(response['Body'].read())
        return result[0]["generated_text"]


# Example 1: Simple LLM Chain
def example_simple_chain():
    """Simple question-answering chain"""
    llm = SageMakerLLM(
        endpoint_name="vllm-llama3-8b",
        model_kwargs={"max_tokens": 256, "temperature": 0.7}
    )

    template = """Question: {question}

Answer: Let me think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain
    response = chain.run("What is the capital of France?")
    print(f"Response: {response}")


# Example 2: Conversation with Memory
def example_conversation_chain():
    """Conversational chain with memory"""
    llm = SageMakerLLM(endpoint_name="vllm-llama3-8b")

    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

    # Multi-turn conversation
    response1 = conversation.predict(input="Hi, my name is Alice")
    print(f"Response 1: {response1}")

    response2 = conversation.predict(input="What's my name?")
    print(f"Response 2: {response2}")  # Should remember "Alice"


# Example 3: Agent with Tools
def example_agent_with_tools():
    """LangChain agent with custom tools"""

    def search_tool(query: str) -> str:
        """Mock search tool"""
        return f"Search results for: {query}"

    def calculator_tool(expression: str) -> str:
        """Calculator tool"""
        try:
            return str(eval(expression))
        except:
            return "Invalid expression"

    tools = [
        Tool(name="Search", func=search_tool, description="Search for information"),
        Tool(name="Calculator", func=calculator_tool, description="Calculate math expressions"),
    ]

    llm = SageMakerLLM(endpoint_name="vllm-llama3-8b", model_kwargs={"temperature": 0})

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    response = agent.run("What is 25 * 4 + 10?")
    print(f"Agent response: {response}")


# Example 4: Structured Output
def example_structured_output():
    """Parse structured output using Pydantic"""

    class Person(BaseModel):
        name: str = Field(description="Person's name")
        age: int = Field(description="Person's age")
        occupation: str = Field(description="Person's occupation")

    parser = PydanticOutputParser(pydantic_object=Person)

    template = """Extract person information from the text.

{format_instructions}

Text: {text}

Output:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    llm = SageMakerLLM(endpoint_name="vllm-llama3-8b")
    chain = LLMChain(llm=llm, prompt=prompt)

    text = "Alice is a 30-year-old software engineer working at Amazon."
    output = chain.run(text=text)

    person = parser.parse(output)
    print(f"Parsed: {person}")


if __name__ == "__main__":
    print("Example 1: Simple Chain")
    example_simple_chain()

    print("\nExample 2: Conversation with Memory")
    example_conversation_chain()

    print("\nExample 3: Agent with Tools")
    example_agent_with_tools()

    print("\nExample 4: Structured Output")
    example_structured_output()
