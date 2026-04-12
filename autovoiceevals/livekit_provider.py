"""LiveKit provider — Phase 1 (data messages).

Runs adversarial eval conversations via a LiveKit room using the data
channel (text messages), not audio. The caller bot joins the room,
sends turns as JSON data messages, and waits for the agent to respond
the same way.

Requirements
------------
* pip install "livekit>=1.0.0"
* LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env

Agent requirements
------------------
The target agent must listen on the configured data_topic and reply
on the same topic. Message format (JSON):

    Caller → Agent:  {"role": "system",    "content": "<system prompt>"}  # optional, if inject_system_prompt=True
    Caller → Agent:  {"role": "user",      "content": "<turn text>"}
    Agent  → Caller: {"role": "assistant", "content": "<reply text>"}

Plain-text responses (non-JSON) are also accepted.

Prompt management
-----------------
* agent_backend="smallest": prompt reads/writes delegate to SmallestClient.
* agent_backend="local": prompt is stored locally (in memory or a file).
  Use inject_system_prompt=True to send the current prompt as the first
  data message of every conversation so the agent can apply it.
* agent_backend="none": get_system_prompt/update_prompt raise
  NotImplementedError — use the livekit provider only for conversations
  and manage prompts externally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

from .models import Turn, Conversation

_log = logging.getLogger(__name__)

DEFAULT_END_PHRASES = [
    "have a great day",
    "goodbye",
    "talk to you soon",
    "take care",
]


class LocalPromptBackend:
    """Manages prompts locally without any external API calls.

    Suitable for self-hosted LiveKit agents that read their prompt from a
    shared file, or for research sessions where the prompt is injected into
    each conversation via the data channel (inject_system_prompt=True).
    """

    def __init__(self, initial_prompt: str = "", prompt_file: str = ""):
        self._file = prompt_file
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file) as f:
                self._prompt = f.read().strip()
        else:
            self._prompt = initial_prompt

    def get_system_prompt(self, agent_id: str) -> str:
        return self._prompt

    def update_prompt(self, agent_id: str, new_prompt: str) -> bool:
        self._prompt = new_prompt
        if self._file:
            with open(self._file, "w") as f:
                f.write(new_prompt)
        return True


class LiveKitClient:
    """Voice platform client that uses LiveKit data channel messages."""

    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        room_prefix: str = "eval",
        data_topic: str = "text",
        response_timeout: float = 30.0,
        agent_join_timeout: float = 30.0,
        end_phrases: list[str] | None = None,
        agent_backend=None,
        inject_system_prompt: bool = False,
    ):
        """
        Args:
            url:                  LiveKit server WebSocket URL (wss://...).
            api_key:              LiveKit API key.
            api_secret:           LiveKit API secret.
            room_prefix:          Prefix for generated room names.
            data_topic:           Data channel topic for messages.
            response_timeout:     Seconds to wait for each agent response.
            agent_join_timeout:   Seconds to wait for the agent to join.
            end_phrases:          Phrases that signal conversation end.
            agent_backend:        Optional client for prompt management
                                  (e.g. SmallestClient, LocalPromptBackend).
                                  If None, prompt methods raise NotImplementedError.
            inject_system_prompt: If True, send the current system prompt as a
                                  {"role": "system", ...} data message before the
                                  first caller turn. Requires agent_backend to be set.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_prefix = room_prefix
        self.data_topic = data_topic
        self.response_timeout = response_timeout
        self.agent_join_timeout = agent_join_timeout
        self.end_phrases = end_phrases or DEFAULT_END_PHRASES
        self._backend = agent_backend
        self.inject_system_prompt = inject_system_prompt

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
        dynamic_variables: dict | None = None,
        simulate_timeout_secs: int | None = None,
    ) -> Conversation:
        """Run a multi-turn conversation via LiveKit data messages.

        Each call creates a unique room. The agent is expected to join
        that room (via webhook dispatch or a fixed room name configured
        on the agent side) and respond to data messages on data_topic.

        The ``scenario``, ``dynamic_variables``, and ``simulate_timeout_secs``
        parameters are accepted for interface compatibility but are unused in
        data-channel mode.
        """
        return asyncio.run(
            self._run_async(assistant_id, scenario_id, caller_turns, max_turns)
        )

    async def _run_async(
        self,
        assistant_id: str,
        scenario_id: str,
        caller_turns: list[str],
        max_turns: int,
    ) -> Conversation:
        try:
            from livekit import rtc
            from livekit.api import AccessToken, VideoGrants
        except ImportError:
            conv = Conversation(scenario_id=scenario_id)
            conv.error = (
                "livekit package not installed. "
                "Run: pip install 'livekit>=1.0.0'"
            )
            return conv

        conv = Conversation(scenario_id=scenario_id)
        total_latency = 0.0

        # Unique room per conversation to avoid cross-talk
        room_name = f"{self.room_prefix}-{scenario_id}-{uuid.uuid4().hex[:8]}"
        identity = f"caller-{uuid.uuid4().hex[:6]}"

        token = (
            AccessToken(self.api_key, self.api_secret)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        room = rtc.Room()
        response_queue: asyncio.Queue[str] = asyncio.Queue()
        agent_joined = asyncio.Event()
        loop = asyncio.get_running_loop()

        @room.on("data_received")
        def on_data(packet):
            # livekit >= 1.0: packet is a DataPacket with .data bytes attribute
            try:
                raw = bytes(packet.data) if hasattr(packet, "data") else bytes(packet)
                text = raw.decode("utf-8")

                # Filter by topic: only accept messages on our configured topic
                topic = getattr(packet, "topic", None)
                if topic is not None and topic != self.data_topic:
                    _log.debug("data_received: ignoring topic=%s", topic)
                    return

                # Ignore messages we sent ourselves (guard for edge cases)
                sender = getattr(packet, "participant", None)
                if sender is not None and getattr(sender, "identity", "") == identity:
                    return

                _log.debug("data_received: topic=%s len=%d", topic, len(raw))
                # Use call_soon_threadsafe in case the SDK fires from a non-asyncio thread
                loop.call_soon_threadsafe(response_queue.put_nowait, text)
            except Exception as exc:
                _log.debug("data_received decode error: %s", exc)

        @room.on("participant_connected")
        def on_participant(_participant):
            agent_joined.set()

        try:
            await room.connect(self.url, token)
        except Exception as e:
            conv.error = f"LiveKit connect failed: {str(e)[:200]}"
            return conv

        # Agent may already be in the room when we join
        if room.remote_participants:
            agent_joined.set()

        try:
            await asyncio.wait_for(
                agent_joined.wait(), timeout=self.agent_join_timeout
            )
        except asyncio.TimeoutError:
            conv.error = (
                f"Agent did not join room '{room_name}' "
                f"within {self.agent_join_timeout}s. "
                "Ensure the agent is configured to dispatch to this room."
            )
            await room.disconnect()
            return conv

        # Optionally inject the current system prompt before any caller turns
        if self.inject_system_prompt and self._backend is not None:
            prompt = self._backend.get_system_prompt(assistant_id)
            sys_payload = json.dumps({"role": "system", "content": prompt}).encode("utf-8")
            try:
                await room.local_participant.publish_data(
                    sys_payload, reliable=True, topic=self.data_topic
                )
                await asyncio.sleep(0.2)  # give agent time to apply the prompt
            except Exception as e:
                _log.debug("Failed to inject system prompt: %s", e)

        # Run turns
        for msg in caller_turns[:max_turns]:
            if not msg or not msg.strip():
                msg = "..."

            conv.turns.append(Turn(role="caller", content=msg))

            payload = json.dumps({"role": "user", "content": msg}).encode("utf-8")

            try:
                # Drain any stale data from previous turns before starting the timer
                while not response_queue.empty():
                    response_queue.get_nowait()

                t0 = time.time()
                await room.local_participant.publish_data(
                    payload,
                    reliable=True,
                    topic=self.data_topic,
                )

                raw_response = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=self.response_timeout,
                )
                latency = (time.time() - t0) * 1000

                # Accept JSON {"role": "assistant", "content": "..."} or plain text
                try:
                    parsed = json.loads(raw_response)
                    agent_msg = parsed.get("content", raw_response)
                except (json.JSONDecodeError, AttributeError):
                    agent_msg = raw_response

                conv.turns.append(
                    Turn(role="assistant", content=agent_msg, latency_ms=latency)
                )
                total_latency += latency

                if any(p in agent_msg.lower() for p in self.end_phrases):
                    break

            except asyncio.TimeoutError:
                conv.error = f"Response timeout (>{self.response_timeout}s)"
                break
            except Exception as e:
                conv.error = str(e)[:200]
                break

            await asyncio.sleep(0.1)

        await room.disconnect()

        n = len(conv.agent_turns)
        conv.avg_latency_ms = total_latency / n if n else 0
        return conv

    # ------------------------------------------------------------------
    # Prompt management (delegated to agent_backend)
    # ------------------------------------------------------------------

    def get_system_prompt(self, agent_id: str) -> str:
        """Read the current system prompt via the configured backend."""
        if self._backend is not None:
            return self._backend.get_system_prompt(agent_id)
        raise NotImplementedError(
            "No agent_backend configured for LiveKit provider. "
            "Set livekit.agent_backend to 'local' (with a system_prompt) or "
            "'smallest' in config.yaml, or manage prompts externally."
        )

    def update_prompt(self, agent_id: str, new_prompt: str) -> bool:
        """Update the system prompt via the configured backend."""
        if self._backend is not None:
            return self._backend.update_prompt(agent_id, new_prompt)
        raise NotImplementedError(
            "No agent_backend configured for LiveKit provider. "
            "Set livekit.agent_backend to 'local' (with a system_prompt) or "
            "'smallest' in config.yaml, or manage prompts externally."
        )
