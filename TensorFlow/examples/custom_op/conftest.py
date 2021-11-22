###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import pytest

def pytest_addoption(parser):
    parser.addoption("--custom_op_lib", action="store", help="Location of CustomOp lib", required=True)

@pytest.fixture
def custom_op_lib_path(request):
    return request.config.getoption('--custom_op_lib')
