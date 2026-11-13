# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes's model-gated system-prompt guidance, added per tier.

Hermes picks these blocks by the chat's model name, which behind Switchyard is switchyard/<preset>
and matches none. The tap adds them to each tier's draft and serving calls as Hermes would for that
tier's model, right after STEER_CHANNEL_NOTE, where Hermes's _guidance_parts puts them.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Mapping, MutableMapping, Optional, Set, Tuple


def _blocks(model: str, tool_names: Optional[Set[str]]) -> List[str]:
    from agent.prompt_builder import (
        EXECUTION_GUIDANCE_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE, TOOL_USE_ENFORCEMENT_GUIDANCE,
        TOOL_USE_ENFORCEMENT_MODELS, execution_guidance_text,
    )
    from agent.system_prompt import _model_gate
    from hermes_cli.config import load_config_readonly

    agent = (load_config_readonly() or {}).get("agent")
    agent = agent if isinstance(agent, Mapping) else {}
    blocks: List[str] = []
    if _model_gate(agent.get("tool_use_enforcement", "auto"), model, TOOL_USE_ENFORCEMENT_MODELS):
        blocks.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
        if any(g in model.lower() for g in ("gemini", "gemma")):
            blocks.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
    if _model_gate(agent.get("execution_guidance", "auto"), model, EXECUTION_GUIDANCE_MODELS):
        blocks.append(execution_guidance_text(tool_names))
    return [block.strip() for block in blocks if block and block.strip()]


def _text_slots(body: MutableMapping[str, Any]) -> Iterator[Tuple[MutableMapping[str, Any], str]]:
    """Where a system prompt's text sits: Chat Completions, Anthropic Messages and Responses requests."""
    def parts(holder: MutableMapping[str, Any], key: str) -> Iterator[Tuple[MutableMapping[str, Any], str]]:
        value = holder.get(key)
        if isinstance(value, str):
            yield holder, key
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    yield part, "text"

    for message in body.get("messages") or ():
        if isinstance(message, dict) and message.get("role") in ("system", "developer"):
            yield from parts(message, "content")
    yield from parts(body, "system")
    yield from parts(body, "instructions")


def _tool_names(body: Mapping[str, Any]) -> Set[str]:
    names = set()
    for tool in body.get("tools") or ():
        if isinstance(tool, dict):
            function = tool.get("function")
            name = function.get("name") if isinstance(function, dict) else tool.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def add_tier_guidance(body: MutableMapping[str, Any]) -> bool:
    """Adds the guidance Hermes gives ``body["model"]`` to its system prompt; False leaves ``body`` untouched."""
    from agent.prompt_builder import STEER_CHANNEL_NOTE

    anchor = STEER_CHANNEL_NOTE.strip()
    slots = [(holder, key) for holder, key in _text_slots(body) if anchor in holder[key]]
    if len(slots) != 1 or slots[0][0][slots[0][1]].count(anchor) != 1:
        return False
    holder, key = slots[0]
    blocks = [block for block in _blocks(str(body.get("model") or ""), _tool_names(body)) if block not in holder[key]]
    if not blocks:
        return False
    holder[key] = holder[key].replace(anchor, "\n\n".join([anchor, *blocks]), 1)
    return True
