import sys
import os

import unittest
import numpy as np

sys.path.append(os.path.abspath(os.getcwd()))
# noinspection PyUnresolvedReferences
import gf   # noqa: E402
# noinspection PyUnresolvedReferences
import bch  # noqa: E402


def A_(*args, **kwargs):
    kwargs['dtype'] = np.int_
    return np.asarray(*args, **kwargs)


class NumpyTest(unittest.TestCase):
    def shortDescription(self):
        return None

    def assertNdarrayEqual(self, value, correct, msg=None):
        msg = msg or '\n\nExpected equal ndarrays, received:\nvalue={}\ncorrect={}\n\n'.format(
            repr(value), repr(correct))
        self.assertIsInstance(value, (np.ndarray, np.generic),
                              msg=msg + 'Object value is not an instance of np.ndarray. type={}'.format(type(value)))
        self.assertIsInstance(correct, (np.ndarray, np.generic),
                              msg=msg + 'Object correct is not an instance of np.ndarray. type={}'.format(
                                  type(correct)) + '\n\nTHERE IS AN ERROR IN THE TESTS, PLEASE, '
                                                   'CONSIDER SUBMITTING AN ISSUE TO OUR GITHUB PAGE\n\n')
        self.assertTrue(np.can_cast(value.dtype, correct.dtype, casting='same_kind') and
                        np.can_cast(correct.dtype, value.dtype, casting='same_kind'),
                        msg=msg + 'The dtype of np.ndarray can\'t be safely converted. dtypes=({}, {})'.format(
                            value.dtype, correct.dtype))
        diff = None
        try:
            np.testing.assert_array_equal(value, correct, verbose=False)
        except AssertionError as e:
            diff = msg + str(e)
        if diff:
            self.fail(diff)
