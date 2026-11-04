# Third-party notices

This repository contains only NVIDIA-authored code, licensed under Apache-2.0
(see [LICENSE](LICENSE)). It bundles, copies and modifies no third-party code,
and it distributes no object code.

The plugin uses the software below at run time. None of it is redistributed
here: each is installed by the user or provided by Hermes Agent, under its own
license.

| Software | License | How the plugin uses it |
|---|---|---|
| [nemo-switchyard](https://github.com/NVIDIA-NeMo/Switchyard) (NVIDIA) | Apache-2.0 | Routing engine, declared in `switchyard/plugin.yaml` and installed from PyPI. The Switchyard project distributes it, including the third-party code its wheels link, under its own release. |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent), including Hermes Desktop | MIT | Host that loads the plugin. |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | Dashboard API, from Hermes Agent's environment. |
| [HTTPX](https://github.com/encode/httpx) | BSD-3-Clause | HTTP client for the sidecar, from Hermes Agent's environment. |
| [Pydantic](https://github.com/pydantic/pydantic) | MIT | Dashboard API data validation, from Hermes Agent's environment. |
| [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) | BSD-3-Clause | Routing notices in Hermes's CLI, from Hermes Agent's environment. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Optional local model server, run as a separate process over local HTTP. |
