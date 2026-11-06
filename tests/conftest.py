import pytest

from tests.support import CHAT, SYSTEMONE, registry


@pytest.fixture(autouse=True)
def trajectory_store(tmp_path, monkeypatch):
    """Keep tests out of the shared data/trajectory store."""
    store = tmp_path / "trajectory-store"
    monkeypatch.setenv("TRAJECTORY_STORE", str(store))
    return store


@pytest.fixture(autouse=True)
def models(tmp_path, monkeypatch):
    """Resolve test models from a test registry; the packaged one names live endpoints."""
    path = registry(
        tmp_path,
        {
            "test-model": {"chat_completions": CHAT},
            "typesafe-ai/jev": {"systemone": SYSTEMONE},
        },
    )
    monkeypatch.setenv("MODELS_FILE", str(path))
    monkeypatch.setenv("TEST_API_KEY", "smoke-test-secret")
    for role in ("SOLVER", "VERIFIER"):
        monkeypatch.delenv(role + "_REASONING_EFFORT", raising=False)
    return path
