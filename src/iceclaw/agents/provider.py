"""Model providers behind one interface: the real OpenAI client and a scripted fake.

Both go through Microsoft Agent Framework's real function-invocation loop. Every underlying model
round-trip (``_inner_get_response``) is counted against the run budget and stored verbatim
(request messages, response, token usage) so the original interaction is auditable. API keys are
never placed in messages, logs or stored records.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    Message,
)
from agent_framework.openai import OpenAIChatClient

from iceclaw.config import Settings
from iceclaw.swarm.runtime import RunLimits, SwarmRuntime

# 2 430 base64url-style characters (contains 'sk-' mid-token on purpose); TEST FIXTURE ONLY
OPAQUE_FIXTURE = "Qr7_x-9Yk" * 270


def measure_request_chars(request: list[Any], options: Any) -> int:
    """Characters of everything the provider will tokenize: messages, instructions, tool schemas."""
    total = len(json.dumps(request, default=str))
    for key, value in dict(options or {}).items():
        if value is None:
            continue
        if key == "tools":
            for t in value:
                d = t.to_dict() if hasattr(t, "to_dict") else t
                total += len(json.dumps(d, default=str))
        else:
            total += len(json.dumps(value, default=str))
    return total


class _Instrumented:
    """Mixin placed FIRST in the MRO: meters each real model call below MAF's tool loop."""

    _runtime: SwarmRuntime
    _agent_id: str

    def bind(self, runtime: SwarmRuntime, agent_id: str) -> None:
        self._runtime, self._agent_id = runtime, agent_id

    def _inner_get_response(self, *, messages, stream, options, **kwargs):  # type: ignore[override]
        if stream:
            raise NotImplementedError("IceClaw agents use non-streaming model calls")
        nxt = super()._inner_get_response  # type: ignore[misc]
        runtime, agent_id = self._runtime, self._agent_id

        async def call() -> ChatResponse:
            request = [m.to_dict() for m in messages]
            measure = measure_request_chars(request, options)
            # may raise BudgetExhausted BEFORE any spend; otherwise reserves this call's estimate
            rid = runtime.llm_before_call(agent_id, measure)
            started = datetime.now(UTC)
            try:
                resp = await nxt(messages=messages, stream=False, options=options, **kwargs)
            except Exception as e:
                runtime.llm_record(
                    agent_id,
                    started=started,
                    request=request,
                    response=None,
                    error=f"{type(e).__name__}: {e}"[:600],
                    usage={},
                    reservation=rid,
                    measure_chars=measure,
                )
                raise
            except BaseException:  # cancellation / timeout: never leak a reservation
                runtime.llm_release(rid)
                raise
            runtime.llm_record(
                agent_id,
                started=started,
                request=request,
                response=resp.to_dict(),
                error=None,
                usage=dict(resp.usage_details or {}),
                reservation=rid,
                measure_chars=measure,
            )
            return resp

        return call()


class InstrumentedOpenAIChatClient(_Instrumented, OpenAIChatClient):
    """The real provider (OpenAI Responses API through ``agent-framework-openai``)."""


def build_openai_client(
    settings: Settings, runtime: SwarmRuntime, agent_id: str, limits: RunLimits
) -> InstrumentedOpenAIChatClient:
    """Real client from IceClaw runtime settings. Raises ``ConfigError`` (e.g. MISSING_API_KEY)
    before anything is constructed; the model ID is never guessed."""
    settings.require_live_llm()
    client = InstrumentedOpenAIChatClient(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        function_invocation_configuration={
            "max_iterations": limits.max_roundtrips_per_turn,
            "max_function_calls": limits.max_function_calls_per_turn,
        },
        # MAF reads a .env file only when a path is passed explicitly; we never pass one.
    )
    client.bind(runtime, agent_id)
    return client


# ------------------------------------------------------------------------- scripted fake
@dataclass
class ScriptContext:
    """What a scripted policy may look at (it is the fake model's only input)."""

    agent_id: str
    roundtrip: int  # 0-based count of this client's model calls so far
    messages: list[Message]
    last_user_text: str
    last_tool_results: list[str]

    def text(self) -> str:
        return " ".join(m.text or "" for m in self.messages)


ScriptReply = str | list[tuple[str, dict[str, Any]]]
Policy = Callable[[ScriptContext], ScriptReply]


class _ScriptedBase(BaseChatClient):
    """Holds the scripted ``_inner_get_response``; initialised by ``ScriptedChatClient``."""

    _policy: Policy
    _script_agent: str
    _n: int
    prompts: list[str]  # every prompt this fake saw (for secrecy tests)
    tool_results: list[str]  # every tool result MAF fed back to this fake
    latency_s: float = 0.0  # simulated provider latency (tests of in-flight concurrency)
    # when True, add an opaque protected_data reasoning item, like the real provider
    emit_reasoning: bool = False

    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        async def call() -> ChatResponse:
            if self.latency_s > 0:
                await asyncio.sleep(self.latency_s)
            msgs = list(messages)
            users = [m.text or "" for m in msgs if m.role == "user"]
            results = [
                str(c.result)
                for m in msgs
                if m.role == "tool"
                for c in m.contents
                if c.type == "function_result"
            ]
            self.tool_results.extend(r for r in results if r not in self.tool_results)
            self.prompts.append("\n".join(m.text or "" for m in msgs))
            ctx = ScriptContext(
                self._script_agent, self._n, msgs, users[-1] if users else "", results
            )
            reply = self._policy(ctx)
            self._n += 1
            if isinstance(reply, str):
                contents = [Content.from_text(reply)]
            else:
                contents = [
                    Content.from_function_call(
                        call_id=f"call-{self._script_agent}-{self._n}-{i}",
                        name=name,
                        arguments=args,
                    )
                    for i, (name, args) in enumerate(reply)
                ]
            if self.emit_reasoning:
                rid = f"rs_{self._script_agent}_{self._n}"
                reasoning = Content.from_text_reasoning(
                    id=rid, text="", protected_data=OPAQUE_FIXTURE
                )
                contents.insert(0, reasoning)
            # like a real provider, count everything tokenized: messages + instructions + tools
            chars_in = measure_request_chars([m.to_dict() for m in msgs], options)
            return ChatResponse(
                messages=[Message(role="assistant", contents=contents)],
                response_id=f"scripted-{self._script_agent}-{self._n}",
                usage_details={
                    "input_token_count": chars_in // 4 + 1,
                    "output_token_count": 10,
                    "total_token_count": chars_in // 4 + 11,
                },
            )

        return call()


class ScriptedChatClient(
    _Instrumented, FunctionInvocationLayer, ChatMiddlewareLayer, _ScriptedBase
):
    """Deterministic stand-in for a model: same MAF function loop, same metering/recording."""

    def __init__(self, policy: Policy, agent_id: str, **fic: Any) -> None:
        super().__init__(function_invocation_configuration=fic) if fic else super().__init__()
        self._policy, self._script_agent, self._n = policy, agent_id, 0
        self.prompts, self.tool_results = [], []
