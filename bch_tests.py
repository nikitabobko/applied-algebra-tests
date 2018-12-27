import random
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from tqdm import tqdm

import gf
from bch import BCH


class Tests(unittest.TestCase):
    def test_correct_1(self):
        self.do_correctness_test(2, 1)

    def test_correct_2(self):
        self.do_correctness_test(5, 10)

    def test_correct_3(self):
        self.do_correctness_test(8, 60)

    def test_correct_4(self):
        self.do_correctness_test(9, 60)  # about 30 seconds

    def test_correct_5(self):
        self.do_correctness_test(9, 20)  # about 8 seconds

    def test_correct_6(self):
        self.do_correctness_test(16, 1)  # about 128 seconds

    def test_correct_7(self):
        self.do_correctness_test(9, 60, get_msg=lambda len: np.concatenate([[0, 1], np.zeros(len - 2).astype(int)]))

    # def test_correct_zhopa_polnaya_ne_doschitaetsya(self):
    #     self.do_correctness_test(16, 32767)  # infinity...
    #     self.fail(msg="I don't believe that your algorithm is fast enough to get to this point")

    def do_correctness_test(self, q, t, get_msg=None):
        pbar = tqdm(total=8)
        n = 2 ** q - 1
        code = BCH(n, t)
        pbar.update(1)
        self.assertTrue(np.logical_or((code.g == [0]), (code.g == [1])).all())
        self.assertRaises(BaseException, lambda it: code.decode([np.zeros(n + 1).astype(int)]))
        pbar.update(1)
        x = np.zeros(n + 1).astype(int)
        x[0] = 1
        x[-1] = 1
        assert_array_equal(gf.polydiv(x, code.g, code.pm)[1], [0])
        pbar.update(1)
        m = gf.polydeg(code.g)
        k = n - m
        self.assertRaises(BaseException, lambda it: code.encode([np.zeros(k + 1).astype(int)]))
        if get_msg is None:
            msg = [random.randint(0, 1) for _ in range(0, k)]
        else:
            msg = get_msg(k)
        coded = code.encode([msg])[0]
        assert_array_equal(coded[0:k], msg)
        pbar.update(1)
        self.assertTrue((gf.polyval(coded, code.R, code.pm) == [0]).all())
        pbar.update(1)
        assert_array_equal(gf.polydiv(coded, code.g, code.pm)[1], [0])
        pbar.update(1)
        assert_array_equal(code.decode([coded])[0], coded)
        pbar.update(1)
        if t >= 1:
            broken = np.array(coded)
            goig_to_make_count_mistakes = random.randint(0, t)
            mistakes = 0
            i = 0
            while mistakes < goig_to_make_count_mistakes and i < len(broken):
                if random.randint(0, 1) == 1:
                    broken[i] = 1 - broken[i]
                    mistakes += 1
                i += 1
            assert_array_equal(code.decode([broken], method='pgz')[0], coded)
            assert_array_equal(code.decode([broken], method='euclid')[0], coded)
            assert_array_equal(code.decode([broken])[0], coded)
        pbar.update(1)
        pbar.close()


if __name__ == '__main__':
    unittest.main()
