import httpx
import pytest

from assistants import (
    Completion,
    FileRAGSystem,
    HybridMemoryAssistant,
    OpenRouterClient,
    ProviderError,
    SupportAssistant,
)


class FakeClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return Completion("Grounded answer", total_tokens=12)


def test_support_policy_is_system_message_and_empty_input_rejected():
    client = FakeClient()
    assistant = SupportAssistant(client)
    assert assistant.invoke({"user_input": "Help"}).content == "Grounded answer"
    assert client.calls[0][0]["role"] == "system"
    with pytest.raises(ValueError):
        assistant.invoke({"user_input": " "})


def test_memory_is_bounded_and_failed_calls_do_not_mutate_it():
    client = FakeClient()
    assistant = HybridMemoryAssistant(client, max_turns=2)
    for text in ["a", "b", "c"]:
        assistant.chat(text)
    assert len(assistant.messages) == 4
    assert assistant.messages[0]["content"] == "b"
    assistant.clear_memory()
    assert assistant.get_memory_info()["turns"] == 0


def test_rag_ingestion_is_idempotent_persisted_and_retrieval_grounded(tmp_path):
    path = tmp_path / "chunks.json"
    rag = FileRAGSystem(path, FakeClient())
    texts = ["Python is a programming language.", "A database stores structured records."]
    assert rag.add_documents(texts, [{"source": "python"}, {"source": "database"}]) == 2
    assert rag.add_documents(texts, [{"source": "python"}, {"source": "database"}]) == 0
    assert FileRAGSystem(path).search("programming")[0].source == "python"
    assert rag.ask("programming")["sources"] == ["python"]
    assert rag.ask("unrelatedxyz")["sources"] == []


def test_provider_structured_usage_and_http_failure():
    client = OpenRouterClient(
        "fake-key",
        "fake-model",
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 7}},
            )
        ),
    )
    assert client.complete([{"role": "user", "content": "hi"}]).total_tokens == 7
    broken = OpenRouterClient(
        "fake-key", "fake-model", httpx.MockTransport(lambda request: httpx.Response(429))
    )
    with pytest.raises(ProviderError, match="429"):
        broken.complete([])
