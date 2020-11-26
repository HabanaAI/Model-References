import pytest


def pytest_report_header(config):
    return "project stereo"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--local", action="store_true", default=False, help="run local or on sagemaker"
    )


@pytest.fixture
def local(request):
    return request.config.getoption("--local")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    skip_skip = pytest.mark.skip(reason="test marked skip")
    for item in items:
        if "skip" in item.keywords:
            item.add_marker(skip_skip)
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
