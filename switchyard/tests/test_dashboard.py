# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starting Hermes's web dashboard for the "Open in browser" button."""

from __future__ import annotations

import hermes_cli

from plugins.switchyard import sidecar


def _desktop_backend_env(monkeypatch, tmp_path):
    for key in sidecar._DESKTOP_BACKEND_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("HERMES_WEB_DIST", str(tmp_path / "missing" / "web_dist"))
    package = tmp_path / "package" / "hermes_cli"
    package.mkdir(parents=True)
    monkeypatch.setattr(hermes_cli, "__file__", str(package / "__init__.py"))
    return package / "web_dist"


def test_the_dashboard_drops_the_desktop_backends_identity_and_uses_the_bundled_web_dist(monkeypatch, tmp_path):
    bundled = _desktop_backend_env(monkeypatch, tmp_path)
    bundled.mkdir()
    (bundled / "index.html").write_text("<html></html>")

    env = sidecar._dashboard_env()

    assert not set(sidecar._DESKTOP_BACKEND_ENV) & env.keys()
    assert env["HERMES_WEB_DIST"] == str(bundled.resolve())


def test_a_missing_web_dist_is_not_passed_on_when_there_is_no_bundled_one(monkeypatch, tmp_path):
    _desktop_backend_env(monkeypatch, tmp_path)

    assert "HERMES_WEB_DIST" not in sidecar._dashboard_env()


def test_a_working_web_dist_is_kept(monkeypatch, tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "index.html").write_text("<html></html>")
    monkeypatch.setenv("HERMES_WEB_DIST", str(custom))

    assert sidecar._dashboard_env()["HERMES_WEB_DIST"] == str(custom)
