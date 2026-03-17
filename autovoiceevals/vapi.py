"""Vapi API client.

Handles conversation simulation (Chat API) and assistant management
(GET/PATCH). Class-based — no global state.
"""

from __future__ import annotations

import time

import requests

from .models import Turn, Conversation

DEFAULT_BASE_URL = "https://api.vapi.ai"

# Phrases that signal the agent is ending the conversation.
DEFAULT_END_PHRASES = [
    "have a great day",
    "goodbye",
    "talk to you soon",
    "take care",
]


class VapiClient:
    """Client for the Vapi REST API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        end_phrases: list[str] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.end_phrases = end_phrases or DEFAULT_END_PHRASES
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def run_conversation(
        self,
        assistant_id: str,
        scenario_id: str,
        caller_turns: list[str],
        max_turns: int = 12,
        scenario=None,
        dynamic_variables=None,
    ) -> Conversation:
        """Run a multi-turn conversation via the Vapi Chat API."""
        conv = Conversation(scenario_id=scenario_id)
        prev_chat_id = None
        total_latency = 0.0

        for msg in caller_turns[:max_turns]:
            if not msg or not msg.strip():
                msg = "..."
            conv.turns.append(Turn(role="caller", content=msg))

            body: dict = {"assistantId": assistant_id, "input": msg}
            if prev_chat_id:
                body["previousChatId"] = prev_chat_id

            try:
                t0 = time.time()
                resp = requests.post(
                    f"{self.base_url}/chat",
                    headers=self._headers,
                    json=body,
                    timeout=30,
                )
                latency = (time.time() - t0) * 1000

                if resp.status_code not in (200, 201):
                    conv.error = f"API {resp.status_code}: {resp.text[:200]}"
                    break

                data = resp.json()
                prev_chat_id = data.get("id", prev_chat_id)

                agent_msg = ""
                if data.get("output"):
                    agent_msg = data["output"][-1].get("content", "")

                conv.turns.append(
                    Turn(role="assistant", content=agent_msg, latency_ms=latency)
                )
                total_latency += latency
                conv.total_cost += data.get("cost", 0.0)

                if any(p in agent_msg.lower() for p in self.end_phrases):
                    break

            except requests.exceptions.Timeout:
                conv.error = "Timeout (>30s)"
                break
            except Exception as e:
                conv.error = str(e)[:200]
                break

            time.sleep(0.1)

        n = len(conv.agent_turns)
        conv.avg_latency_ms = total_latency / n if n else 0
        return conv

    # ------------------------------------------------------------------
    # Assistant management
    # ------------------------------------------------------------------

    def get_assistant(self, assistant_id: str) -> dict:
        """Fetch full assistant configuration."""
        resp = requests.get(
            f"{self.base_url}/assistant/{assistant_id}",
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()

    def get_system_prompt(self, assistant_id: str) -> str:
        """Read the current system prompt."""
        data = self.get_assistant(assistant_id)
        return data["model"]["messages"][0]["content"]

    def update_prompt(self, assistant_id: str, new_prompt: str) -> bool:
        """Update the assistant's system prompt.

        Must send the full model object (model + provider + messages)
        because Vapi silently ignores partial updates.

        Returns True on success.
        """
        data = self.get_assistant(assistant_id)
        model_cfg = data.get("model", {})

        resp = requests.patch(
            f"{self.base_url}/assistant/{assistant_id}",
            headers=self._headers,
            json={
                "model": {
                    "model": model_cfg.get("model", "gpt-4o-mini"),
                    "provider": model_cfg.get("provider", "openai"),
                    "messages": [{"role": "system", "content": new_prompt}],
                }
            },
        )
        return resp.status_code in (200, 201)
