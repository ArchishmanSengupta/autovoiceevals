"""Smallest AI (Atoms) client.

Handles agent prompt management via REST API and conversation
simulation using Claude + the agent's system prompt.

Prompt management:
  - GET  /agent/{id}/workflow       → read system prompt
  - PATCH /workflow/{workflowId}    → update system prompt

Conversations:
  Atoms agents only accept audio input via LiveKit rooms (no text chat
  API). To run adversarial eval conversations, we simulate them using
  Claude as the agent model with the actual system prompt from the
  platform. This tests the prompt itself — which is the artifact
  autoresearch optimizes.
"""

from __future__ import annotations

import time

import requests

from .models import Turn, Conversation

ATOMS_BASE_URL = "https://api.smallest.ai/atoms/v1"

END_PHRASES = [
    "have a great day", "goodbye", "talk to you soon", "take care",
]


class SmallestClient:
    """Client for Smallest AI Atoms platform."""

    def __init__(self, api_key: str, llm_client=None):
        """
        Args:
            api_key: Smallest AI API key.
            llm_client: LLMClient instance for simulated conversations.
                        Required for run_conversation().
        """
        self.api_key = api_key
        self.llm = llm_client
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> dict:
        """Fetch full agent configuration."""
        resp = requests.get(
            f"{ATOMS_BASE_URL}/agent/{agent_id}",
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def get_system_prompt(self, agent_id: str) -> str:
        """Read the current system prompt from the agent's workflow."""
        resp = requests.get(
            f"{ATOMS_BASE_URL}/agent/{agent_id}/workflow",
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["data"]["prompt"]

    def update_prompt(self, agent_id: str, new_prompt: str) -> bool:
        """Update the agent's system prompt.

        For single_prompt agents, updates via the workflow PATCH endpoint.
        Preserves existing tools configuration.
        Returns True on success.
        """
        agent = self.get_agent(agent_id)
        workflow_id = agent["workflowId"]

        # Preserve existing tools
        workflow_resp = requests.get(
            f"{ATOMS_BASE_URL}/agent/{agent_id}/workflow",
            headers=self._headers,
        )
        workflow_resp.raise_for_status()
        existing_tools = workflow_resp.json()["data"].get("tools", [])

        resp = requests.patch(
            f"{ATOMS_BASE_URL}/workflow/{workflow_id}",
            headers=self._headers,
            json={
                "type": "single_prompt",
                "singlePromptConfig": {
                    "prompt": new_prompt,
                    "tools": existing_tools,
                },
            },
        )
        return resp.status_code == 200

    # ------------------------------------------------------------------
    # Conversation (simulated via Claude)
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
        """Run a simulated multi-turn conversation.

        Uses the agent's actual system prompt (fetched from the platform)
        with Claude as the responding model. This tests the prompt quality
        directly, which is the artifact autoresearch optimizes.
        """
        if self.llm is None:
            raise RuntimeError(
                "SmallestClient requires an LLMClient for conversations. "
                "Pass llm_client= to the constructor."
            )

        conv = Conversation(scenario_id=scenario_id)
        total_latency = 0.0

        # Fetch the current system prompt from the platform
        try:
            system_prompt = self.get_system_prompt(assistant_id)
        except Exception as e:
            conv.error = f"Failed to fetch prompt: {str(e)[:200]}"
            return conv

        # Build conversation using Claude as the agent
        messages: list[dict] = []

        for msg in caller_turns[:max_turns]:
            if not msg or not msg.strip():
                msg = "..."

            conv.turns.append(Turn(role="caller", content=msg))
            messages.append({"role": "user", "content": msg})

            try:
                t0 = time.time()
                agent_response = self.llm.chat(
                    system=system_prompt,
                    messages=messages,
                    max_tokens=500,
                )
                latency = (time.time() - t0) * 1000
            except Exception as e:
                conv.error = f"LLM error: {str(e)[:200]}"
                break

            conv.turns.append(
                Turn(role="assistant", content=agent_response, latency_ms=latency)
            )
            messages.append({"role": "assistant", "content": agent_response})
            total_latency += latency

            # Check for conversation end
            if any(p in agent_response.lower() for p in END_PHRASES):
                break

            time.sleep(0.1)

        n = len(conv.agent_turns)
        conv.avg_latency_ms = total_latency / n if n else 0
        return conv
