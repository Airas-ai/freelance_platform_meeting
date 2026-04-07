"""
Tests for test_agent.py (conversational booking API).

Run: pytest tests/test_agent_api.py -v
Skip LLM/DB integration without DEEPSEEK_API_KEY: tests marked @pytest.mark.integration
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from test_agent import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_describes_messages_endpoint():
    r = client.get("/")
    assert r.status_code == 200
    assert "messages" in r.json().get("input", "")


def test_appointment_requires_messages():
    r = client.post("/agent/appointment", json={"messages": []})
    assert r.status_code == 422


@pytest.mark.integration
def test_appointment_calls_llm():
    """Needs DEEPSEEK_API_KEY and network."""
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    r = client.post(
        "/agent/appointment",
        json={
            "messages": [
                {"role": "user", "content": "Hello, I want to book a meeting."},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "assistant_message" in data
    assert "gathered" in data


@pytest.mark.integration
@pytest.mark.mysql
def test_full_flow_no_conflict_needs_db_and_llm():
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    if not all(os.getenv(k) for k in ("MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE")):
        pytest.skip("MYSQL_* not set")
    r = client.post(
        "/agent/appointment",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Book a meeting. My name is Test User, email test@example.com, "
                        "date 2038-01-20 from 10:00 to 11:00."
                    ),
                },
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("complete") in (True, False)
