# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model-gated guidance the tap adds to each tier's system prompt."""

from __future__ import annotations

import copy

from plugins.switchyard import guidance


def test_a_tier_gets_the_system_prompt_hermes_builds_for_its_own_model():
    from agent.prompt_builder import (
        PARALLEL_TOOL_CALL_GUIDANCE, STEER_CHANNEL_NOTE, TASK_COMPLETION_GUIDANCE, TOOL_USE_ENFORCEMENT_GUIDANCE,
        execution_guidance_text,
    )
    from agent.system_prompt import _join_tier

    names = {"web_search", "web_extract"}
    head, tail = [TASK_COMPLETION_GUIDANCE, PARALLEL_TOOL_CALL_GUIDANCE, STEER_CHANNEL_NOTE], ["Model: switchyard/vgr"]
    direct = _join_tier([*head, TOOL_USE_ENFORCEMENT_GUIDANCE, execution_guidance_text(names), *tail])

    def request(model, *parts):
        return {"model": model, "tools": [{"type": "function", "function": {"name": n}} for n in sorted(names)],
                "messages": [{"role": "system", "content": _join_tier([*parts, *tail])}, {"role": "user", "content": "hi"}]}

    qwen = request("Qwen3.8-27B-UD-Q4_K_M", *head)
    assert guidance.add_tier_guidance(qwen) and qwen["messages"][0]["content"] == direct
    # Hermes gives Claude none, and a prompt without the anchor (a one-shot's) is left as it came.
    for body in (request("azure/anthropic/claude-opus-5-5", *head), request("Qwen3.8-27B-UD-Q4_K_M", *head[:2])):
        before = copy.deepcopy(body)
        assert not guidance.add_tier_guidance(body) and body == before
