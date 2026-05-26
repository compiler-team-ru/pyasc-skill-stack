import pytest
from asc.runtime import config


def pytest_addoption(parser):
    parser.addoption("--backend", type=config.Backend, default=config.Backend.Model)
    parser.addoption("--platform", type=config.Platform, default=config.Platform.Ascend950PR_9599)


@pytest.fixture
def backend(request):
    return request.config.getoption("--backend")


@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")