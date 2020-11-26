def test_get_non_default_args():
    def func(a, b, c=2):
        return a + b - c
    from stereo.common.general_utils import get_non_default_args
    assert get_non_default_args(func) == ['a', 'b']
