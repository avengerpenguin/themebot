import pathlib
from typing import Generator

import pytest
from filelock import FileLock
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models import BaseLLM
from testcontainers.core.container import DockerContainer

OLLAMA_PORT = 11434
MODELS = {
    "openchat",
    "llama3",
}


def pytest_addoption(parser):
    parser.addoption("--url", action="store")


def start_ollama(url_file: pathlib.Path = None) -> Generator[str, None, None]:
    data_dir = pathlib.Path(__file__).parent / ".ollama"
    with DockerContainer("ollama/ollama:latest").with_exposed_ports(
        OLLAMA_PORT
    ).with_volume_mapping(str(data_dir), "/root/.ollama", "rw") as container:
        lib_dir = data_dir / "models" / "manifests" / "registry.ollama.ai" / "library"

        if lib_dir.exists():
            installed_models = {str(p.name) for p in lib_dir.iterdir()}
        else:
            installed_models = set()

        for model in installed_models - MODELS:
            container.exec(f"ollama rm {model}")

        for model in MODELS - installed_models:
            container.exec(f"ollama pull {model}")

        url = f"http://localhost:{container.get_exposed_port(11434)}"
        if url_file:
            url_file.write_text(url)
        yield url


@pytest.fixture(scope="session")
def ollama_url(request, tmp_path_factory, worker_id) -> Generator[str, None, None]:
    url = request.config.option.url
    if url:
        yield url
    elif worker_id == "master":
        yield from start_ollama()
    else:
        root_tmp_dir = tmp_path_factory.getbasetemp().parent
        url_file = root_tmp_dir / "url.txt"
        with FileLock(str(url_file) + ".lock"):
            if url_file.is_file():
                yield url_file.read_text()
            else:
                yield from start_ollama(url_file)


@pytest.fixture(params=sorted(MODELS), scope="session")
def llm(request, ollama_url: str) -> BaseLLM:
    return Ollama(model=request.param, base_url=ollama_url)
