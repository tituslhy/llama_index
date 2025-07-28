from collections.abc import AsyncGenerator

from acp_sdk import Message
from acp_sdk.server import Server

from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentStream


class AgentServer:
    def __init__(self, agents: list[FunctionAgent | ReActAgent]):
        self.agents = agents
        self.server = Server()

    def serve(self):
        agent = self.agents[0]  # Select the first agent, or customize selection logic

        @self.server.agent()
        async def llamaindex_rag_agent(input: list[Message]) -> AsyncGenerator:
            """LlamaIndex agent that answers questions using the Docling knowledge base. The agent answers questions in streaming mode."""
            query = str(input[-1])
            handler = agent.run(query)
            async for ev in handler.stream_events():
                if isinstance(ev, AgentStream):
                    yield ev.delta
            response = await handler
            yield str(response)

        self.server.run()
