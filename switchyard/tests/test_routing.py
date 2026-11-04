# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end routing through the plugin's sidecar.

Requests travel Hermes-side client -> tap -> Switchyard's in-process route for
the requested preset -> tap -> mock tiers, so the stages the monitor reports are
the calls the real route made.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

pytest.importorskip("switchyard_rust.server")

from plugins.switchyard import settings as sy_settings  # noqa: E402
from plugins.switchyard import escalation, providers, sidecar, stages, tap  # noqa: E402
from plugins.switchyard.tap import SELECTED_MODEL_HEADER  # noqa: E402

from .mock_tiers import MockTier  # noqa: E402

TOKEN = "test-front-token"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def vgr(monkeypatch, request):
    local = MockTier("local", "The answer is 4.")
    cloud = MockTier("cloud", "Cloud says 4.")
    monkeypatch.setenv(sy_settings.TOKEN_ENV, TOKEN)
    monkeypatch.setenv("SWITCHYARD_TEST_CLOUD_KEY", "cloud-secret")
    local_tier = {"base_url": local.base_url, "model": "mock/local"}
    cloud_model = "mock/cloud"
    if getattr(request, "param", None) == "claude":
        cloud_model = sy_settings.DEFAULT_CLOUD_MODEL
    if getattr(request, "param", None) == "managed":
        # The configured base_url is a dead port: managed traffic must follow the resolver.
        local_tier = {"base_url": f"http://127.0.0.1:{_free_port()}/v1", "model": "mock/local", "runtime": "managed"}
        monkeypatch.setattr(sidecar, "managed_local_endpoint", lambda: (local.base_url, "managed-key"))
    cloud_tier = {"base_url": cloud.base_url, "model": cloud_model, "api_key_env": "SWITCHYARD_TEST_CLOUD_KEY"}
    if getattr(request, "param", None) in ("provider", "codex"):
        # A Hermes provider tier: endpoint and token come from Hermes's resolver on each call.
        codex = request.param == "codex"
        provider = "openai-codex" if codex else "mock-provider"
        cloud_tier = {"source": "provider", "provider": provider, "model": "mock/cloud",
                      **({"format": "openai_responses"} if codex else {})}
        cloud.stream_only = codex
        monkeypatch.setattr(providers, "_runtime", lambda tier: {
            "provider": provider, "api_mode": "codex_responses" if codex else "chat_completions",
            "base_url": cloud.base_url, "api_key": "provider-token",
        })
    cfg = sy_settings.load({
        "port": _free_port(),
        "local": local_tier,
        "cloud": cloud_tier,
        "presets": {"vgr": {"algorithm": "vgr"}},
    })
    thread = threading.Thread(target=sidecar.serve, args=(cfg,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not (sidecar.health(cfg) or {}).get("switchyard"):
        assert time.monotonic() < deadline, "sidecar did not start"
        time.sleep(0.1)
    client = httpx.Client(base_url=cfg.front_base_url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60)
    yield cfg, client, local, cloud
    client.close()
    sidecar.stop(cfg)
    thread.join(timeout=10)
    local.stop()
    cloud.stop()


CLOUD_ONLY = next(model for model, tier in sy_settings.PIN_MODELS.items() if tier == "cloud")
CLOUD_PIN = sy_settings.pin_preset("cloud").name


def _ask(client: httpx.Client, text: str, model: str = "switchyard/vgr", **extra) -> httpx.Response:
    return client.post("/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": text}],
        **extra,
    })


def test_verified_answer_is_served_by_the_local_tier(vgr):
    cfg, client, local, cloud = vgr
    response = _ask(client, "What is 2+2?")
    assert response.status_code == 200
    assert response.headers[SELECTED_MODEL_HEADER] == cfg.local.model
    assert response.json()["choices"][0]["message"]["content"] == local.answer
    assert not any(stages.classify("cloud", r["body"]) == stages.CLOUD_SERVE for r in cloud.requests)

    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert turn["served_tier"] == "local"
    assert turn["outcome"] == "local_committed"
    assert turn["stages"][0]["stage"] == stages.LOCAL_DRAFT
    assert {s["stage"] for s in turn["stages"]} >= {stages.TYPING, stages.READOUT}
    assert turn["tokens"]["local"]["prompt"] == 50 * turn["tokens"]["local"]["calls"]


@pytest.mark.parametrize("vgr", ["managed"], indirect=True)
def test_managed_local_tier_uses_the_resolved_endpoint_and_key(vgr):
    cfg, client, local, _ = vgr
    response = _ask(client, "What is 2+2?")
    assert response.status_code == 200
    assert response.headers[SELECTED_MODEL_HEADER] == cfg.local.model
    assert local.requests
    assert all(r["authorization"] == "Bearer managed-key" for r in local.requests)


def test_refuted_answer_escalates_to_the_cloud_tier(vgr):
    cfg, client, _, cloud = vgr
    response = _ask(client, "What is 2+2? REFUTE")
    assert response.status_code == 200
    assert response.headers[SELECTED_MODEL_HEADER] == cfg.cloud.model
    assert response.json()["choices"][0]["message"]["content"] == cloud.answer
    assert all(r["authorization"] == "Bearer cloud-secret" for r in cloud.requests)

    state = sidecar.fetch_state(cfg)
    turn = state["turns"][0]
    assert turn["outcome"] == "escalated_after_verification"
    assert turn["stages"][-1]["stage"] == stages.CLOUD_SERVE
    assert state["totals"]["served"]["cloud"] == 1
    assert state["totals"]["cloud"]["calls"] == sum(1 for s in turn["stages"] if s["tier"] == "cloud")


@pytest.mark.parametrize("vgr", ["claude"], indirect=True)
def test_claude_cloud_tier_is_called_over_anthropic_messages(vgr):
    cfg, client, _, cloud = vgr
    response = _ask(client, "What is 2+2? REFUTE")
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == cloud.answer
    assert cloud.requests
    for call in cloud.requests:
        assert call["path"] == "/v1/messages"
        assert call["x_api_key"] == "cloud-secret"
    serve = [c["body"] for c in cloud.requests if stages.classify("cloud", c["body"]) == stages.CLOUD_SERVE]
    assert len(serve) == 1
    assert serve[0]["thinking"] == {"type": "adaptive"}
    assert serve[0]["output_config"] == {"effort": "high"}

    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert turn["served_tier"] == "cloud"
    assert turn["stages"][-1]["stage"] == stages.CLOUD_SERVE
    assert turn["tokens"]["cloud"]["prompt"] == 50 * turn["tokens"]["cloud"]["calls"]


@pytest.mark.parametrize("vgr", ["provider"], indirect=True)
def test_provider_tier_is_called_with_the_resolved_token_not_the_callers(vgr):
    cfg, client, _, cloud = vgr
    response = _ask(client, "What is 2+2? REFUTE")
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == cloud.answer
    assert cloud.requests
    assert all(r["authorization"] == "Bearer provider-token" for r in cloud.requests)
    assert sidecar.fetch_state(cfg)["turns"][0]["served_tier"] == "cloud"


@pytest.mark.parametrize("vgr", ["codex"], indirect=True)
def test_codex_tier_is_called_over_the_responses_api(vgr):
    cfg, client, _, cloud = vgr
    response = _ask(client, "What is 2+2?", model=CLOUD_ONLY)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == cloud.answer
    assert cloud.requests
    for call in cloud.requests:
        assert call["path"].endswith("/responses")
        assert call["authorization"] == "Bearer provider-token"
        assert call["originator"]
        assert call["body"]["store"] is False
        assert call["body"]["stream"] is True
    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert turn["served_tier"] == "cloud"
    assert turn["tokens"]["cloud"]["prompt"] == 50


@pytest.mark.parametrize("vgr", ["codex"], indirect=True)
def test_vgr_checks_and_serves_on_a_codex_tier_that_only_streams(vgr):
    cfg, client, _, cloud = vgr
    response = _ask(client, "What is 2+2? REFUTE")
    assert response.status_code == 200, response.text
    assert response.headers[SELECTED_MODEL_HEADER] == cfg.cloud.model
    assert response.json()["choices"][0]["message"]["content"] == cloud.answer
    assert cloud.requests and all(call["body"]["stream"] is True for call in cloud.requests)
    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert turn["outcome"] == "escalated_after_verification"
    assert {s["status"] for s in turn["stages"] if s["tier"] == "cloud"} == {"ok"}


def test_pin_sends_every_request_to_one_tier_until_released(vgr):
    cfg, client, local, cloud = vgr

    def pin(tier: str) -> None:
        assert client.post(f"{cfg.tap_url}/monitor/pin", json={"pin": tier}).json()["pin"] == ("" if tier == "off" else tier)

    pin("cloud")
    pinned = _ask(client, "What is 2+2?")
    assert pinned.headers[SELECTED_MODEL_HEADER] == cfg.cloud.model
    assert pinned.json()["choices"][0]["message"]["content"] == cloud.answer
    assert local.requests == []

    pin("local")
    assert _ask(client, "What is 3+3? REFUTE").headers[SELECTED_MODEL_HEADER] == cfg.local.model
    assert len(cloud.requests) == 1

    pin("off")
    assert _ask(client, "What is 4+4? REFUTE").headers[SELECTED_MODEL_HEADER] == cfg.cloud.model
    state = sidecar.fetch_state(cfg)
    assert state["pin"] == ""
    first, second, third = state["turns"][::-1]
    assert (first["outcome"], second["outcome"]) == ("served_cloud", "served_local")
    assert third["outcome"] == "escalated_after_verification"
    assert client.post(f"{cfg.tap_url}/monitor/pin", json={"pin": "both"}).status_code == 400


def test_the_pickers_single_tier_models_skip_routing_and_are_no_escalation(vgr):
    cfg, client, local, cloud = vgr
    spec = importlib.util.spec_from_file_location(
        "switchyard_provider_profile", Path(__file__).parents[1] / "provider" / "__init__.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    assert set(profile._SINGLE_TIER_MODELS) == set(sy_settings.PIN_MODELS)

    def ask(text: str, model: str) -> httpx.Response:
        return client.post("/chat/completions", headers={"x-switchyard-session-id": model},
                           json={"model": model, "messages": [{"role": "user", "content": text}]})

    assert ask("What is 2+2?", "switchyard/only-capable").headers[SELECTED_MODEL_HEADER] == cfg.cloud.model
    assert local.requests == [] and len(cloud.requests) == 1
    assert ask("What is 3+3? REFUTE", "switchyard/only-efficient").headers[SELECTED_MODEL_HEADER] == cfg.local.model
    assert len(cloud.requests) == 1

    state = sidecar.fetch_state(cfg)
    assert [t["preset"] for t in state["turns"]] == ["pin-local", "pin-cloud"]
    assert escalation.since(state, "switchyard/only-capable", 0)["changes"] == []


def test_the_turn_counts_a_streaming_drafts_tokens_before_it_ends(vgr):
    cfg, client, local, _ = vgr
    local.midstream = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        reply = pool.submit(_ask, client, "What is 2+2?", stream=True)
        deadline = time.monotonic() + 15
        while not (current := (sidecar.fetch_state(cfg) or {}).get("current") or {}).get("tokens", {}).get("completion"):
            assert time.monotonic() < deadline, "no tokens counted while the draft streamed"
            time.sleep(0.05)
        local.midstream.set()
        assert reply.result(timeout=30).status_code == 200

    draft = next(s for s in current["turn"]["stages"] if s["stage"] == stages.LOCAL_DRAFT)
    # The draft's one content event so far, at four characters a token; its prompt, estimated until
    # the usage comes, from the request's text.
    assert (draft["status"], draft["completion_tokens"]) == ("running", -(-len(local.answer) // 4))
    assert draft["prompt_tokens"] > 0
    assert current["tokens"]["completion"] >= draft["completion_tokens"]
    assert current["tokens"]["prompt"] >= draft["prompt_tokens"]
    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert turn["tokens"]["local"]["prompt"] == 50 * turn["tokens"]["local"]["calls"]
    finished = next(s for s in turn["stages"] if s["stage"] == stages.LOCAL_DRAFT)
    assert (finished["completion_tokens"], finished["estimated"]) == (max(1, len(local.answer) // 4), False)


def test_side_tasks_go_straight_to_the_local_tier(vgr):
    cfg, client, local, cloud = vgr
    hermes = {tap.CLIENT_HEADER: tap.HERMES_CLIENT}

    def served(content, headers, model="switchyard/vgr") -> str:
        response = client.post("/chat/completions", headers=headers,
                               json={"model": model, "messages": [{"role": "user", "content": content}]})
        assert response.status_code == 200
        return response.headers[SELECTED_MODEL_HEADER]

    assert served("Title this chat. REFUTE", hermes) == cfg.local.model
    assert len(local.requests) == 1 and cloud.requests == []
    image = [{"type": "text", "text": "Describe this."},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    assert served(image, hermes) == cfg.cloud.model
    assert served("What is 2+2? REFUTE", {**hermes, tap.SESSION_HEADER: "s1"}) == cfg.cloud.model
    assert served("What is 3+3? REFUTE", {}) == cfg.cloud.model
    assert served("Title this chat.", hermes, CLOUD_ONLY) == cfg.cloud.model
    assert sidecar.fetch_state(cfg)["turns"][-1]["preset"] == sy_settings.pin_preset("local").name


def test_concurrent_turns_keep_their_own_stages(vgr):
    cfg, client, _, _ = vgr
    with ThreadPoolExecutor(max_workers=2) as pool:
        refuted = pool.submit(_ask, client, "What is 2+2? REFUTE")
        time.sleep(0.05)
        verified = pool.submit(_ask, client, "What is 3+3?")
        assert refuted.result().status_code == verified.result().status_code == 200

    turns = {t["preview"]: t for t in sidecar.fetch_state(cfg)["turns"]}
    escalated, local = turns["What is 2+2? REFUTE"], turns["What is 3+3?"]
    assert escalated["outcome"] == "escalated_after_verification"
    assert [s["stage"] for s in escalated["stages"]].count(stages.CLOUD_SERVE) == 1
    assert local["outcome"] == "local_committed"
    assert all(s["tier"] == "local" for s in local["stages"])


def test_a_turn_records_hermes_request_id(vgr):
    cfg, client, _, _ = vgr
    response = client.post("/chat/completions", headers={"x-hermes-request-id": "t1:api:1"}, json={
        "model": "switchyard/vgr", "messages": [{"role": "user", "content": "What is 2+2?"}],
    })
    assert response.status_code == 200
    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert (turn["hermes_request_id"], turn["id"]) == ("t1:api:1", 1)
    assert turn["uid"] and turn["run_id"]


def test_a_caller_that_hangs_up_cancels_the_local_draft(vgr):
    cfg, _, local, cloud = vgr
    local.hold = 30.0
    started = time.monotonic()
    with httpx.Client(base_url=cfg.front_base_url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=1) as impatient:
        with pytest.raises(httpx.ReadTimeout):
            _ask(impatient, "What is 2+2?")
    assert local.abandoned.wait(10), "the local draft kept running after the caller left"
    assert time.monotonic() - started < 10
    deadline = time.monotonic() + 5
    while (turn := sidecar.fetch_state(cfg)["turns"][0])["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert turn["status"] == "http_499"
    assert not any(stages.classify("cloud", r["body"]) == stages.CLOUD_SERVE for r in cloud.requests)


def test_a_draft_stream_left_open_after_its_end_is_not_recorded_as_cancelled(vgr):
    cfg, client, local, _ = vgr
    local.linger = 1.0  # Switchyard stops reading at [DONE] and hangs up before the stream closes
    response = _ask(client, "What is 2+2?", stream=True)
    assert response.status_code == 200
    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert (turn["status"], turn["outcome"]) == ("ok", "local_committed")
    assert {s["stage"]: s["status"] for s in turn["stages"]}[stages.LOCAL_DRAFT] == "ok"


def test_a_cloud_reply_that_goes_silent_is_given_up_on(vgr, monkeypatch):
    cfg, client, _, cloud = vgr
    monkeypatch.setattr(tap, "_CLOUD_TIMEOUT", httpx.Timeout(connect=15.0, read=1.0, write=60.0, pool=None))
    cloud.stall = 30.0
    started = time.monotonic()
    with client.stream("POST", "/chat/completions", json={
        "model": CLOUD_ONLY, "messages": [{"role": "user", "content": "What is 2+2?"}], "stream": True,
    }) as response:
        body = "".join(response.iter_text())
    assert time.monotonic() - started < 15
    assert cloud.abandoned.wait(5), "the stalled cloud stream was never shut"
    assert cloud.answer not in body
    assert len(cloud.requests) == 1, "Switchyard's retries waited out the stall again"


def test_a_stalled_cloud_is_waited_on_once_and_the_local_model_answers(vgr, monkeypatch):
    cfg, client, local, cloud = vgr
    monkeypatch.setattr(tap, "_CLOUD_TIMEOUT", httpx.Timeout(connect=15.0, read=1.0, write=60.0, pool=None))
    cloud.stall = 30.0
    response = _ask(client, "What is 2+2? REFUTE")
    assert response.status_code == 200, response.text
    assert response.headers[SELECTED_MODEL_HEADER] == cfg.local.model
    assert len(cloud.requests) == 1
    assert sidecar.fetch_state(cfg)["turns"][0]["cloud_stalled_s"] == 1


def test_each_preset_is_its_own_route_and_ledger(vgr):
    cfg, client, local, cloud = vgr
    assert _ask(client, "What is 2+2?", model=CLOUD_ONLY).json()["choices"][0]["message"]["content"] == cloud.answer
    assert _ask(client, "What is 3+3?", model=sy_settings.DEFAULT_MODEL).status_code == 200
    assert local.requests

    state = sidecar.fetch_state(cfg)
    turns = {t["preview"]: t for t in state["turns"]}
    passthrough, default = turns["What is 2+2?"], turns["What is 3+3?"]
    assert (passthrough["preset"], passthrough["algorithm"]) == (CLOUD_PIN, "passthrough")
    assert passthrough["outcome"] == "served_cloud"
    assert {s["tier"] for s in passthrough["stages"]} == {"cloud"}
    assert (default["preset"], default["outcome"]) == ("vgr", "local_committed")
    assert state["totals"]["by_preset"][CLOUD_PIN]["served"] == {"local": 0, "cloud": 1}
    assert state["totals"]["by_preset"]["vgr"]["served"] == {"local": 1, "cloud": 0}


def test_unknown_preset_is_rejected_before_routing(vgr):
    _, client, local, cloud = vgr
    assert _ask(client, "hi", model="switchyard/missing").status_code == 404
    assert local.requests == cloud.requests == []


def test_streamed_turn_is_relayed_and_accounted(vgr):
    cfg, client, local, _ = vgr
    with client.stream("POST", "/chat/completions", json={
        "model": sy_settings.DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert local.answer in body
    turn = sidecar.fetch_state(cfg)["turns"][0]
    assert turn["stream"] is True
    assert turn["served_tier"] == "local"


def test_a_failed_turn_names_the_tool_calls_that_already_ran(vgr):
    _, client, _, cloud = vgr
    cloud.status = 401
    response = client.post("/chat/completions", json={"model": "switchyard/vgr", "messages": [
        {"role": "user", "content": "Tidy the folder. REFUTE"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "terminal", "arguments": '{"command": "rm -f a.txt"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": ""},
    ]})
    assert response.status_code == 401
    message = response.json()["error"]["message"]
    assert message.startswith(
        "Before this error, 1 tool call for this message had already run, and nothing was undone: terminal. ")
    assert "rm -f" not in message


_NO_CREDIT = json.dumps({"error": {
    "message": "You exceeded your current quota, please check your plan and billing details.",
    "type": "insufficient_quota", "code": "insufficient_quota"}})


@pytest.mark.parametrize("refused", ["cloud", "local"])
def test_a_tier_out_of_credits_hands_its_steps_to_the_other_tier_without_retries(vgr, refused):
    cfg, client, local, cloud = vgr
    tier = cloud if refused == "cloud" else local
    tier.status, tier.error_body = 429, _NO_CREDIT
    served = cfg.local.model if refused == "cloud" else cfg.cloud.model
    task = {"role": "user", "content": "List the files, then summarize. REFUTE"}
    call = {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
    steps = [[task], [task, {"role": "assistant", "content": "", "tool_calls": [call]},
                      {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}]]

    for index, messages in enumerate(steps):
        response = client.post("/chat/completions", headers={
            "x-switchyard-session-id": "s1", "x-initiator": "agent" if index else "user",
        }, json={"model": "switchyard/vgr", "messages": messages})
        assert response.status_code == 200, response.text
        assert response.headers[SELECTED_MODEL_HEADER] == served

    state = sidecar.fetch_state(cfg)
    assert state["credit_alerts"][refused]["message"].startswith("You exceeded your current quota")
    assert state["turns"][-1]["out_of_credits"]["tier"] == refused
    for turn in state["turns"]:
        refusals = [s for s in turn["stages"] if s["tier"] == refused]
        # Each refused call was tried once: Switchyard retries a 429, so the tap passes it on as a 403.
        assert len({s["stage"] for s in refusals}) == len(refusals)
        assert {s["status"] for s in refusals} <= {"out_of_credits"}


def test_a_turn_both_tiers_fail_tells_hermes_which_ran_out_of_credits(vgr):
    _, client, local, cloud = vgr
    cloud.status, cloud.error_body = 429, _NO_CREDIT
    local.status = 500
    response = _ask(client, "What is 2+2? REFUTE")
    assert response.status_code == 402, response.text
    message = response.json()["error"]["message"]
    assert "capable model is out of credits (insufficient_quota), and no other model answered" in message
    assert message.count("You exceeded your current quota") == 1


def test_a_turn_both_tiers_refuse_for_lack_of_credit_names_both(vgr):
    _, client, local, cloud = vgr
    for tier in (local, cloud):
        tier.status, tier.error_body = 429, _NO_CREDIT
    response = _ask(client, "What is 2+2?")
    assert response.status_code == 402, response.text
    assert "Both of Switchyard's models, efficient and capable, are out of credits" in response.json()["error"]["message"]


def test_front_door_rejects_unauthenticated_and_browser_callers(vgr):
    cfg, _, local, _ = vgr
    payload = {"model": sy_settings.DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}]}
    url = f"{cfg.front_base_url}/chat/completions"
    assert httpx.post(url, json=payload).status_code == 401
    assert httpx.post(url, json=payload, headers={
        "Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example",
    }).status_code == 403
    assert httpx.get(f"{cfg.tap_url}/monitor/state").status_code == 401
    assert local.requests == []
