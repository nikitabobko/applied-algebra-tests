import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
import gf

import unittest
import numpy as np


class Tests(unittest.TestCase):
    def setUp(self):
        self.basedir = os.path.realpath(os.path.dirname(__file__))
        self.pow_matrices = {}
        with np.load(os.path.join(self.basedir, 'pow_matrices.npz')) as pms:
            for primpoly, pm in pms.iteritems():
                self.pow_matrices[int(primpoly)] = pm

    def assertNdarrayEqual(self, n1, n2, msg=None):
        msg = msg or '\n\nExpected equal ndarrays n1 and n2, recieved:\nn1={}\nn2={}\n\n'.format(repr(n1), repr(n2))
        self.assertIsInstance(n1, (np.ndarray, np.generic), msg=msg+'Object n1 is not an instance of np.ndarray. type={}'.format(type(n1)))
        self.assertIsInstance(n2, (np.ndarray, np.generic), msg=msg+'Object n2 is not an instance of np.ndarray. type={}'.format(type(n2)))
        self.assertTrue(np.can_cast(n1.dtype, n2.dtype, casting='same_kind') and
                        np.can_cast(n2.dtype, n1.dtype, casting='same_kind'),
                        msg=msg+'The dtype of np.ndarray can\'t be safely converted. dtypes=({}, {})'.format(
                            n1.dtype, n2.dtype))
        diff = None
        try:
            np.testing.assert_array_equal(n1, n2, verbose=False)
        except AssertionError as e:
            diff = msg+str(e)
        if diff:
            self.fail(diff)

    def test_gen_pow_matrix(self):
        for primpoly in self.pow_matrices.keys():
            with self.subTest(primpoly=primpoly):
                self.assertNdarrayEqual(self.pow_matrices[primpoly], gf.gen_pow_matrix(primpoly))

    _arithmetic_tests = [
        # primpoly, a, b, (a*b, a/b, a+b, sum(a, axis=0), sum(a, axis=1), ..., sum(a, axis=-1))
        (19, np.asarray(13), np.asarray(1), (np.asarray(13), np.asarray(13), np.asarray(12), np.asarray(13))),
        (59, np.arange(10), 10-np.arange(10), (
            np.asarray([0, 9, 16, 9, 24, 17, 24, 9, 16, 9]),
            np.asarray([0, 15, 19, 8, 23, 1, 28, 20, 4, 9]),
            np.asarray([10, 8, 10, 4, 2, 0, 2, 4, 10, 8]),
            np.asarray(1))),
        (130207,
         np.asarray([[90186, 79514, 13029, 119929],
                     [96050, 129315, 83614, 23044],
                     [84395, 69827, 122226, 1384]]),
         np.asarray([[109294, 65478, 929, 12126],
                     [46610, 35400, 91278, 74123],
                     [100775, 116919, 57796, 25605]]),
         (np.asarray([[42267, 62999, 37217, 35944],
                      [53485, 16640, 20032, 26237],
                      [46388, 53025, 24813, 6823]]),
          np.asarray([[34410, 5025, 28777, 7080],
                      [5827, 26296, 12404, 6633],
                      [48337, 35291, 29326, 19722]]),
          np.asarray([[51876, 117084, 12612, 129831],
                      [114976, 95083, 8720, 97167],
                      [49164, 55412, 81078, 24941]]),
          np.asarray([89811, 122746, 43273, 101141]),
          np.asarray([110668, 103051, 98674])))
    ]
    _linsolve_tests = [
        # primpoly, A, b, result
        (130207, np.asarray([[64, 23056, 128],
                             [0, 0, 0],
                             [1, 8, 1024]]),
         np.asarray([4, 64, 33128]), np.nan),

        (130207, np.asarray([[64, 23056, 128],
                             [64, 23056, 128],
                             [1, 8, 1024]]),
         np.asarray([4, 64, 33128]), np.nan),

        (108851, np.asarray([[64, 1949, 128],
                             [512, 4, 128],
                             [1, 8, 1024]]),
         np.asarray([4, 64, 48853]), np.asarray([3009, 23136, 63822])),

        (87341, np.asarray([[3272, 59574, 2048, 512],
                            [15319, 54747, 28268, 58909],
                            [59446, 43035, 42843, 56307],
                            [64, 11873, 39430, 27645]]),
         np.asarray([21004, 40721, 20556, 7067]), np.asarray([35048, 24262, 65502, 26384])),

        (19, np.asarray([[3, 7], [12, 1]]), np.asarray([8, 13]), np.asarray([13, 14])),
        (19, np.asarray([[3, 7], [12, 15]]), np.asarray([8, 13]), np.nan),

        (87341, np.asarray([[3272, 59574, 0, 512],
                            [59446, 54747, 0, 58909],
                            [3272, 43035, 0, 56307],
                            [3272, 11873, 0, 27645]]),
         np.asarray([21004, 40721, 7067, 20556]), np.nan),

        (87341, np.asarray([[0, 59574, 2048, 512],
                            [15319, 54747, 28268, 58909],
                            [59446, 43035, 42843, 56307],
                            [64, 11873, 39430, 27645]]),
         np.asarray([21004, 40721, 20556, 7067]), np.asarray([21320, 18899, 5953, 57137])),

        (87341, np.asarray([[1, 59574, 2048, 512],
                            [1, 59574, 28268, 58909],
                            [59446, 43035, 42843, 56307],
                            [64, 11873, 39430, 27645]]),
         np.asarray([21004, 40721, 20556, 7067]), np.asarray([49980, 29479, 12587, 62413]))
    ]

    def test_arithmetic(self):
        for idx, (primpoly, a, b, op_res) in enumerate(self._arithmetic_tests):
            pm = self.pow_matrices[primpoly]
            add_res = op_res[2]
            with self.subTest(idx=idx, op='add'):
                self.assertNdarrayEqual(gf.add(a, b), add_res)
            for op, opname, res in zip((gf.prod, gf.divide), ('prod', 'divide'), op_res[:2]):
                with self.subTest(idx=idx, primpoly=primpoly, op=opname):
                    self.assertNdarrayEqual(op(a, b, pm), res)
            sums = op_res[3:]
            for ax, sum_res in enumerate(sums):
                with self.subTest(idx=idx, op='sum', axis=ax):
                    self.assertNdarrayEqual(gf.sum(a, axis=ax), sum_res)
            sum_last = sums[-1]
            with self.subTest(idx=idx, op='sum', axis=-1):
                self.assertNdarrayEqual(gf.sum(a, axis=-1), sum_last)

    def test_minpoly(self):
        minpoly = gf.minpoly([0b10], self.pow_matrices[0b1011])

        self.assertNdarrayEqual([1, 0, 1, 1], minpoly[0])
        self.assertNdarrayEqual([2, 4, 6], minpoly[1])

    def test_minpoly_2(self):
        minpoly = gf.minpoly([0, 0b10], self.pow_matrices[19])
        self.assertNdarrayEqual(minpoly[0], [1, 0, 0, 1, 1, 0])
        self.assertNdarrayEqual(minpoly[1], [0, 2, 3, 4, 5])

    def test_polyprod(self):
        self.assertNdarrayEqual(gf.polyprod(np.array([1, 0b11]), np.array([1, 0b100]), self.pow_matrices[0b1011]),
                           [1, 0b111, 0b111])

    def test_polyprod_normalize_1(self):
        pm = self.pow_matrices[19]
        p1 = [pm[5, 1], pm[-1, 1]]
        zero = [0]
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), [0])

    def test_polyprod_normalize_2(self):
        pm = self.pow_matrices[19]
        p1 = [pm[-3, 1], pm[-1, 1]]
        zero = [0, 0]
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), [0])

    def test_polyprod_normalize_3(self):
        pm = self.pow_matrices[19]
        p1 = [0, pm[-3, 1], pm[-1, 1]]
        zero = [0]
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), [0])

    def test_polyprod_normalize_4(self):
        pm = self.pow_matrices[19]
        p1 = [0, pm[-3, 1], pm[-1, 1]]
        zero = [0, 0]
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), [0])

    def test_polydiv(self):
        div = gf.polydiv([0b10, 0b1], [0b1], self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], [0b10, 0b1])
        self.assertNdarrayEqual(div[1], [0b0])

    def test_polydiv_normalize(self):
        div = gf.polydiv([0, 0b10, 0b1], [0b1], self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], [0b10, 0b1])
        self.assertNdarrayEqual(div[1], [0b0])

    def test_polydiv_normalize_2(self):
        div = gf.polydiv([0b10, 0b1], [0, 0b1], self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], [0b10, 0b1])
        self.assertNdarrayEqual(div[1], [0b0])

    def test_polydiv_normalize_3(self):
        div = gf.polydiv([0, 0b10, 0b1], [0, 0b1], self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], [0b10, 0b1])
        self.assertNdarrayEqual(div[1], [0b0])

    def test_polydiv_2(self):
        div = gf.polydiv([0b10, 0b1], [0b10], self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], [0b1, 0b101])
        self.assertNdarrayEqual(div[1], [0b0])

    def test_polydiv_3(self):
        div = gf.polydiv([0b10, 0b1], [0b10, 0b0], self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], [0b1])
        self.assertNdarrayEqual(div[1], [0b1])

    def test_polydiv_zero(self):
        pm = self.pow_matrices[5391]
        for elem in pm[:, 1]:
            self.assertRaises(BaseException, lambda: gf.polydiv([elem], [0], pm))

    def test_polyprod_and_polydiv(self):
        pm = self.pow_matrices[108439]
        p1 = [pm[5, 1], pm[3, 1]]
        p2 = [pm[2, 1], pm[-1, 1]]
        self.do_polyprod_and_polydiv_test(p1, p2, pm)

    def test_polyprod_and_polydiv_2(self):
        pm = self.pow_matrices[76553]
        p1 = [pm[5, 1], pm[9, 1]]
        p2 = [pm[6, 1], pm[-2, 1]]
        self.do_polyprod_and_polydiv_test(p1, p2, pm)

    def test_polyadd(self):
        a = gf.polyadd([2, 3], [5, 10, 110])
        b = gf.polyadd([0, 2, 3], [5, 10, 110])
        self.assertNdarrayEqual(a, b)
        self.assertNdarrayEqual([5, 8, 109], b)

    def test_polyadd_2(self):
        a = gf.polyadd([1, 6, 12], [1, 7, 8])
        self.assertNdarrayEqual([1, 4], a)

    def test_polyadd_3(self):
        a = gf.polyadd([1, 2], [1, 2])
        self.assertNdarrayEqual(a, [0])

    def test_divide_zero(self):
        for primpoly in [10187, 104155]:
            pm = self.pow_matrices[primpoly]
            elem = pm[-1, 1]
            with self.subTest(primpoly=primpoly, e1=elem, e2=0):
                self.assertRaises(BaseException, lambda: gf.divide(elem, 0, pm))
            for elem in pm[:, 1]:
                with self.subTest(primpoly=primpoly, e1=0, e2=elem):
                    self.assertEqual(0, gf.divide(0, elem, pm))

    def test_prod_divide(self):
        for primpoly in [19, 59, 357, 54193, 88479]:
            pm = self.pow_matrices[primpoly]
            for elem1 in pm[:357, 1]:
                for elem2 in pm[:357, 1]:
                    with self.subTest(primpoly=primpoly, e1=elem1, e2=elem2):
                        self.assertEqual(elem1, gf.prod(gf.divide(elem1, elem2, pm), elem2, pm))

    def test_euclid(self):
        pm = self.pow_matrices[37]
        p1 = np.array([2, 14, 22, 23, 8, 17, 1, 11, 26, 3])
        p2 = np.array([31, 23, 29, 31, 11, 9])
        max_deg = 3
        result = gf.euclid(p1, p2, pm, max_deg=max_deg)
        self.assertNdarrayEqual(gf.polyadd(gf.polyprod(p1, result[1], pm), gf.polyprod(p2, result[2], pm)), result[0])

    def test_polydeg(self):
        self.assertEqual(0, gf.polydeg([0]))

    def test_polydeg_2(self):
        self.assertEqual(0, gf.polydeg([0, 0]))

    def test_polydeg_3(self):
        self.assertEqual(1, gf.polydeg([1, 1]))

    def test_linsolve(self):
        for idx, (primpoly, A, b, result) in enumerate(self._linsolve_tests):
            with self.subTest(idx=idx, primpoly=primpoly, result=result):
                pm = self.pow_matrices[primpoly]
                check = gf.linsolve(A, b, pm)
                if result is np.nan:
                    self.assertIs(check, np.nan)
                else:
                    self.assertNdarrayEqual(gf.linsolve(A, b, pm), result)

    def test_linsolve_random(self):
        num_tests = 100
        n = 100

        for idx in range(num_tests):
            primpoly = np.random.choice(list(self.pow_matrices.keys()))
            pm = self.pow_matrices[primpoly]
            pm_len = len(pm)
            with self.subTest(idx=idx, primpoly=primpoly):
                solution = np.nan
                while solution is np.nan:
                    A = np.take(pm[:, 1], np.random.randint(pm_len-1, size=(n, n)))
                    b = np.take(pm[:, 1], np.random.randint(pm_len-1, size=n))
                    solution = gf.linsolve(A, b, pm)
                self.assertNdarrayEqual(gf.sum(gf.prod(A, solution, pm), axis=-1), b)

    def do_polyprod_and_polydiv_test(self, p1, p2, pm):
        div = gf.polydiv(p1, p2, pm)
        mult = gf.polyprod(div[0], p2, pm)
        self.assertNdarrayEqual(p1, gf.add(mult, np.concatenate(
            [np.zeros(len(mult) - len(div[1])).astype(int), div[1]])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
