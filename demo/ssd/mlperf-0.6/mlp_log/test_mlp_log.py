"""Test MLPerf logging.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import json
import sys

import pytest

from mlp_log import mlp_log


class TestMLPerfLog(object):
  """Test mlperf log."""

  def test_format(self):
    msg = mlp_log.mlperf_format('foo_key', {'whiz': 'bang'})
    parts = msg.split()
    assert parts[0] == ':::MLL'
    assert float(parts[1]) > 10
    assert parts[2] == 'foo_key:'
    j = json.loads(' '.join(parts[3:]))
    assert j['value'] == {'whiz': 'bang'}
    assert j['metadata']['lineno'] == 21
    assert 'test_mlp_log' in j['metadata']['file']


if __name__ == '__main__':
  sys.exit(pytest.main())
