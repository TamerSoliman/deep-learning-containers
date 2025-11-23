#!/usr/bin/env python3
"""
Streaming Chat Application with vLLM on SageMaker

Features:
- Real-time token streaming
- WebSocket server for frontend
- Conversation history management
- Cost-efficient streaming with vLLM
"""

import boto3
import json
import asyncio
import websockets
from typing import AsyncIterator


class StreamingChatbot:
    """Streaming chatbot using SageMaker vLLM endpoint"""

    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime')
        self.conversation_history = []

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream tokens from vLLM endpoint

        Note: SageMaker streaming requires invoke_endpoint_with_response_stream
        """
        # Build conversation prompt
        full_prompt = self._build_prompt(prompt)

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "stream": True,  # Enable streaming
            }
        }

        # Stream response
        response = self.runtime.invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        # Process stream
        for event in response['Body']:
            if 'PayloadPart' in event:
                chunk = json.loads(event['PayloadPart']['Bytes'].decode())
                if 'token' in chunk:
                    token_text = chunk['token']['text']
                    yield token_text

    def _build_prompt(self, user_message: str) -> str:
        """Build prompt with conversation history"""
        messages = self.conversation_history + [{"role": "user", "content": user_message}]

        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"

        prompt += "Assistant: "
        return prompt

    async def chat(self, user_message: str) -> str:
        """Complete chat turn with streaming"""
        print(f"User: {user_message}")
        print("Assistant: ", end="", flush=True)

        full_response = ""
        async for token in self.stream_response(user_message):
            print(token, end="", flush=True)
            full_response += token

        print()  # Newline

        # Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        return full_response


# WebSocket Server for Frontend Integration
async def websocket_handler(websocket, path, chatbot):
    """Handle WebSocket connections from frontend"""
    async for message in websocket:
        # Receive user message
        data = json.loads(message)
        user_message = data['message']

        # Stream response
        async for token in chatbot.stream_response(user_message):
            # Send each token to frontend
            await websocket.send(json.dumps({"type": "token", "content": token}))

        # Send completion signal
        await websocket.send(json.dumps({"type": "done"}))


async def run_websocket_server(endpoint_name: str, port: int = 8765):
    """Run WebSocket server for chat application"""
    chatbot = StreamingChatbot(endpoint_name)

    async with websockets.serve(
        lambda ws, path: websocket_handler(ws, path, chatbot),
        "localhost",
        port
    ):
        print(f"WebSocket server running on ws://localhost:{port}")
        await asyncio.Future()  # Run forever


# Simple CLI chat interface
async def cli_chat(endpoint_name: str):
    """Command-line chat interface"""
    chatbot = StreamingChatbot(endpoint_name)

    print("Chat started! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['quit', 'exit']:
            break

        await chatbot.chat(user_input)
        print()


if __name__ == "__main__":
    import sys

    endpoint = "vllm-llama3-8b"  # Your SageMaker endpoint

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run WebSocket server
        asyncio.run(run_websocket_server(endpoint))
    else:
        # Run CLI chat
        asyncio.run(cli_chat(endpoint))
