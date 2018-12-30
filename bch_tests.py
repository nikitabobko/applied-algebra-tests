import random
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from tqdm import tqdm

import gf
from bch import BCH


class Tests(unittest.TestCase):
    def test_correct_1(self):
        code = self.do_correctness_test(2, 1, 2)
        self.assertEqual(3, code.dist())

    def test_correct_2(self):
        code = self.do_correctness_test(5, 10, 30)
        self.assertEqual(31, code.dist())

    def test_correct_3(self):
        self.do_correctness_test(8, 60, 246, num_of_msgs=2)

    def test_correct_4(self):
        self.do_correctness_test(9, 60, 408, num_of_msgs=1)

    def test_correct_5(self):
        self.do_correctness_test(9, 20, 171, num_of_msgs=1)

    def test_correct_6(self):
        self.do_correctness_test(9, 60, 408, get_msgs=lambda len: [np.concatenate([[0, 1], np.zeros(len - 2).astype(int)])])

    def test_correct_7(self):

        def foo(it):
            return [[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]]

        self.do_correctness_test(5, 10, 30, get_broken=foo, get_expected_decoded=lambda l: np.array([np.full(l, np.nan)]))

    def test_correct_8(self):

        def broken(it):
            return [[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1]]

        def expected(l):
            return np.array(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 np.full(l, np.nan),
                 np.full(l, np.nan),
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 np.full(l, np.nan),
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 np.full(l, np.nan),
                 np.full(l, np.nan),
                 np.full(l, np.nan)])

        code = self.do_correctness_test(5, 10, 30, get_broken=broken, get_expected_decoded=expected)

    # def test_correct_zhopa_polnaya_ne_doschitaetsya(self):
    #     self.do_correctness_test(16, 32767)  # infinity...
    #     self.fail(msg="I don't believe that your algorithm is fast enough to get to this point")

    def do_correctness_test(self, q, t, deg_g, num_of_msgs=10, get_msgs=None, get_broken=None, get_expected_decoded=None):
        pbar = tqdm(total=7)
        n = 2 ** q - 1
        code = BCH(n, t)
        self.assertEqual(deg_g, np.trim_zeros(code.g, 'f').size - 1)
        pbar.update(1)
        self.assertTrue(np.logical_or((code.g == [0]), (code.g == [1])).all())
        self.assertRaises(BaseException, lambda: code.decode([np.ones(n + 1).astype(int)]))
        pbar.update(1)
        x = np.zeros(n + 1).astype(int)
        x[0] = 1
        x[-1] = 1
        assert_array_equal(gf.polydiv(x, code.g, code.pm)[1], [0])
        pbar.update(1)
        m = gf.polydeg(code.g)
        k = n - m
        self.assertRaises(BaseException, lambda: code.encode([np.ones(k + 1).astype(int)]))
        if get_msgs is None:
            msgs = [[random.randint(0, 1) for _ in range(0, k)] for _ in range(0, num_of_msgs)]
        else:
            msgs = get_msgs(k)
        coded = code.encode(msgs)
        assert_array_equal(coded[:, 0:k], msgs)
        pbar.update(1)
        for i in coded:
            self.assertTrue((gf.polyval(i, code.R, code.pm) == [0]).all())
            assert_array_equal(gf.polydiv(i, code.g, code.pm)[1], [0])
        pbar.update(1)
        assert_array_equal(code.decode(coded), coded)
        pbar.update(1)
        if t >= 1:
            broken = self.break_msgs(coded, t)
            assert_array_equal(code.decode(broken, method='pgz'), coded)
            assert_array_equal(code.decode(broken, method='euclid'), coded)

        if get_broken is not None and get_expected_decoded is not None:
            broken = get_broken(n)
            expected = get_expected_decoded(n)
            self.assert_array_equal(expected, code.decode(broken, method='pgz'))
            self.assert_array_equal(expected, code.decode(broken, method='euclid'))

        pbar.update(1)
        pbar.close()
        return code

    def assert_array_equal(self, expected, actual):
        expected = expected.astype(np.float64)
        actual = actual.astype(np.float64)

        self.assertTrue(((expected == actual) | (np.isnan(expected) & np.isnan(actual))).all(),
                        msg=f"Expected {expected}, Actual: {actual}")

    def break_msgs(self, coded, t):
        broken = np.array(coded)
        for j in range(0, len(broken)):
            elem = broken[j]
            goig_to_make_count_mistakes = random.randint(0, t)
            mistakes = 0
            i = 0
            while mistakes < goig_to_make_count_mistakes and i < len(elem):
                if random.randint(0, 1) == 1:
                    elem[i] = 1 - elem[i]
                    mistakes += 1
                i += 1
            broken[j] = elem
        return broken


if __name__ == '__main__':
    unittest.main()
