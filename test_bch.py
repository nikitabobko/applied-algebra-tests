from common import *


class BCHTests(NumpyTest):
    def test_correct_00(self):
        code = self.do_correctness_test(6, 12, 53)
        self.check_distance(27, code)

    def test_correct_01(self):
        code = self.do_correctness_test(2, 1, 2)
        self.check_distance(3, code)

    def test_correct_02(self):
        code = self.do_correctness_test(5, 10, 30)
        self.check_distance(31, code)

    def test_correct_03(self):
        code = self.do_correctness_test(8, 60, 246, num_of_msgs=2)
        self.check_distance(127, code)

    def test_correct_04(self):
        self.do_correctness_test(9, 60, 408, num_of_msgs=1)

    def test_correct_05(self):
        self.do_correctness_test(9, 20, 171, num_of_msgs=1)

    def test_correct_06(self):
        self.do_correctness_test(9, 60, 408, get_msgs=
            lambda l: A_([np.concatenate([[0, 1], np.zeros(l - 2)])]))

    def test_correct_07(self):
        def broken(_):
            return A_([[0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]])

        def expected(l):
            return A_([np.full(l, np.nan)])

        self.do_correctness_test(5, 10, 30, get_broken=broken, get_expected_decoded=expected)

    def test_correct_08(self):
        def broken(_):
            return A_([[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                       [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                       [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                       [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1]])

        def expected(l):
            return A_([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       np.full(l, np.nan),
                       np.full(l, np.nan),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       np.full(l, np.nan),
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       np.full(l, np.nan),
                       np.full(l, np.nan),
                       np.full(l, np.nan)])

        self.do_correctness_test(5, 10, 30, get_broken=broken, get_expected_decoded=expected)

    def test_correct_09(self):
        code = self.do_correctness_test(9, 120, 501, num_of_msgs=1)
        self.check_distance(255, code)

    def test_correct_10(self):
        self.do_correctness_test(10, 228, 997, num_of_msgs=1)

    def do_correctness_test(self, q, t, deg_g, num_of_msgs=10, get_msgs=None, get_broken=None, get_expected_decoded=None):
        n = 2 ** q - 1
        code = bch.BCH(n, t)
        m = np.trim_zeros(code.g, 'f').size - 1
        k = n - m
        with self.subTest(t='sanity checks'):
            self.assertEqual(deg_g, np.trim_zeros(code.g, 'f').size - 1)
            self.assertTrue(np.logical_or((code.g == [0]), (code.g == [1])).all())
            self.assertRaises(BaseException, lambda: code.decode([np.ones(n + 1).astype(int)]))
            x = np.zeros(n + 1, dtype=np.int)
            x[[0, -1]] = 1
            self.assertNdarrayEqual(gf.polydivmod(x, code.g, code.pm)[1], A_([0]))
            self.assertRaises(BaseException, lambda: code.encode([np.ones(k + 1).astype(int)]))
        if get_msgs is None:
            msgs = np.random.randint(2, size=(num_of_msgs, k))
        else:
            msgs = get_msgs(k)
        coded = code.encode(msgs)
        self.assertNdarrayEqual(coded[:, 0:k], msgs)
        for i in coded:
            with self.subTest(t='encoding', msg_no=i):
                self.assertNdarrayEqual(gf.polyval(i, code.R, code.pm), A_(len(code.R)*(0,)))
                self.assertNdarrayEqual(gf.polydivmod(i, code.g, code.pm)[1], A_([0]))
        with self.subTest(t='decoding without errors', method='euclid (None)'):
            self.assertNdarrayEqual(code.decode(coded), coded)
        with self.subTest(t='decoding without errors', method='pgz'):
            self.assertNdarrayEqual(code.decode(coded, method='pgz'), coded)

        if t >= 1:
            broken = self.break_msgs(coded, t)
            with self.subTest(t='decoding with up to {} errors'.format(t), method='pgz'):
                self.assertNdarrayEqual(code.decode(broken, method='pgz'), coded)
            with self.subTest(t='decoding with up to {} errors'.format(t), method='euclid'):
                self.assertNdarrayEqual(code.decode(broken, method='euclid'), coded)

        if get_broken is not None and get_expected_decoded is not None:
            broken = get_broken(n)
            expected = get_expected_decoded(n)
            with self.subTest(t='decoding hard-coded messages', method='pgz'):
                self.assertNdarrayEqual(code.decode(broken, method='pgz'), expected)
            with self.subTest(t='decoding hard-coded messages', method='euclid'):
                self.assertNdarrayEqual(code.decode(broken, method='euclid'), expected)
        return code

    def break_msgs(self, coded, t):
        broken = coded.copy()
        for j in range(0, len(broken)):
            mistake_num = np.random.randint(t + 1)
            mistake_pos = np.random.permutation(len(broken[j]))[:mistake_num]
            broken[j, mistake_pos] ^= 1
        return broken

    def check_distance(self, expected_distance, code):
        with self.subTest(t='distance check'):
            self.assertEqual(expected_distance, code.dist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
