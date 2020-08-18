###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import pytest

def pytest_addoption(parser):
    parser.addoption("--custom_op_lib", action="store", help="Location of CustomOp lib", required=True)

@pytest.fixture
def custom_op_lib_path(request):
    return request.config.getoption('--custom_op_lib')
