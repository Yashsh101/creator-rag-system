from app.services import langchain_service


class FakeBaseRetriever:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeChain:
    last_kwargs = {}

    @classmethod
    def from_llm(cls, **kwargs):
        cls.last_kwargs = kwargs
        return {"chain": kwargs}


class FakeMemory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class FakePrompt:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_chain_uses_session_memory_and_streaming_llm(monkeypatch):
    monkeypatch.setattr(
        langchain_service,
        "_load_langchain",
        lambda: (FakeBaseRetriever, FakeChatOpenAI, FakeChain, FakeMemory, FakeDocument, FakePrompt),
    )
    langchain_service.clear_session("session-1")

    chain = langchain_service.get_chain("session-1")

    assert chain["chain"]["memory"] is langchain_service._sessions["session-1"]
    assert chain["chain"]["llm"].kwargs["streaming"] is True
    assert chain["chain"]["llm"].kwargs["temperature"] == 0
    assert chain["chain"]["return_source_documents"] is True


def test_clear_session_removes_memory():
    langchain_service._sessions["session-clear"] = object()

    langchain_service.clear_session("session-clear")

    assert "session-clear" not in langchain_service._sessions
