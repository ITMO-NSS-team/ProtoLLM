from langchain_core.messages import BaseMessage, HumanMessage

from protollm.connectors.rest_server import ChatRESTServer


def test_connector():
    conn = ChatRESTServer()
    conn.base_url = 'mock'
    chat = conn.create_chat(messages=[HumanMessage('M1'), HumanMessage('M2'), HumanMessage('M3')])
    assert chat is not None
