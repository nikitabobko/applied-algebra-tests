import random
import unittest

import numpy as np
from numpy.testing import assert_array_equal

import gf


class Tests(unittest.TestCase):
    def test_gen_pow_matrix(self):
        assert_array_equal([[15, 2],
                            [1, 4],
                            [4, 8],
                            [2, 3],
                            [8, 6],
                            [5, 12],
                            [10, 11],
                            [3, 5],
                            [14, 10],
                            [9, 7],
                            [7, 14],
                            [6, 15],
                            [13, 13],
                            [11, 9],
                            [12, 1]], gf.gen_pow_matrix(19))

    def test_gen_pow_matrix_2(self):
        assert_array_equal([[31, 2],
                            [1, 4],
                            [13, 8],
                            [2, 16],
                            [26, 27],
                            [14, 13],
                            [10, 26],
                            [3, 15],
                            [23, 30],
                            [27, 7],
                            [17, 14],
                            [15, 28],
                            [6, 3],
                            [11, 6],
                            [8, 12],
                            [4, 24],
                            [21, 11],
                            [24, 22],
                            [29, 23],
                            [28, 21],
                            [20, 17],
                            [18, 25],
                            [19, 9],
                            [16, 18],
                            [22, 31],
                            [7, 5],
                            [5, 10],
                            [12, 20],
                            [30, 19],
                            [9, 29],
                            [25, 1]], gf.gen_pow_matrix(59))

    def test_minpoly(self):
        minpoly = gf.minpoly([0b10], gf.gen_pow_matrix(0b1011))

        assert_array_equal([1, 0, 1, 1], minpoly[0])
        assert_array_equal([2, 4, 6], minpoly[1])

    def test_minpoly_2(self):
        minpoly = gf.minpoly([0, 0b10], gf.gen_pow_matrix(19))
        assert_array_equal(minpoly[0], [1, 0, 0, 1, 1, 0])
        assert_array_equal(minpoly[1], [0, 2, 3, 4, 5])

    def test_polyprod(self):
        assert_array_equal(gf.polyprod(np.array([1, 0b11]), np.array([1, 0b100]), gf.gen_pow_matrix(0b1011)),
                           [1, 0b111, 0b111])

    def test_polyprod_normalize_1(self):
        pm = gf.gen_pow_matrix(19)
        p1 = [pm[5, 1], pm[-1, 1]]
        zero = [0]
        assert_array_equal(gf.polyprod(p1, zero, pm), [0])

    def test_polyprod_normalize_2(self):
        pm = gf.gen_pow_matrix(19)
        p1 = [pm[-3, 1], pm[-1, 1]]
        zero = [0, 0]
        assert_array_equal(gf.polyprod(p1, zero, pm), [0])

    def test_polyprod_normalize_3(self):
        pm = gf.gen_pow_matrix(19)
        p1 = [0, pm[-3, 1], pm[-1, 1]]
        zero = [0]
        assert_array_equal(gf.polyprod(p1, zero, pm), [0])

    def test_polyprod_normalize_4(self):
        pm = gf.gen_pow_matrix(19)
        p1 = [0, pm[-3, 1], pm[-1, 1]]
        zero = [0, 0]
        assert_array_equal(gf.polyprod(p1, zero, pm), [0])

    def test_polydiv(self):
        div = gf.polydiv([0b10, 0b1], [0b1], gf.gen_pow_matrix(0b1011))
        assert_array_equal(div[0], [0b10, 0b1])
        assert_array_equal(div[1], [0b0])

    def test_polydiv_normalize(self):
        div = gf.polydiv([0, 0b10, 0b1], [0b1], gf.gen_pow_matrix(0b1011))
        assert_array_equal(div[0], [0b10, 0b1])
        assert_array_equal(div[1], [0b0])

    def test_polydiv_normalize_2(self):
        div = gf.polydiv([0b10, 0b1], [0, 0b1], gf.gen_pow_matrix(0b1011))
        assert_array_equal(div[0], [0b10, 0b1])
        assert_array_equal(div[1], [0b0])

    def test_polydiv_normalize_3(self):
        div = gf.polydiv([0, 0b10, 0b1], [0, 0b1], gf.gen_pow_matrix(0b1011))
        assert_array_equal(div[0], [0b10, 0b1])
        assert_array_equal(div[1], [0b0])

    def test_polydiv_2(self):
        div = gf.polydiv([0b10, 0b1], [0b10], gf.gen_pow_matrix(0b1011))
        assert_array_equal(div[0], [0b1, 0b101])
        assert_array_equal(div[1], [0b0])

    def test_polydiv_3(self):
        div = gf.polydiv([0b10, 0b1], [0b10, 0b0], gf.gen_pow_matrix(0b1011))
        assert_array_equal(div[0], [0b1])
        assert_array_equal(div[1], [0b1])

    def test_polydiv_zero(self):
        pm = gf.gen_pow_matrix(5391)
        for elem in pm[:, 1]:
            self.assertRaises(BaseException, lambda: gf.polydiv([elem], [0], pm))

    def test_polyprod_and_polydiv(self):
        pm = gf.gen_pow_matrix(108439)
        p1 = [pm[5, 1], pm[3, 1]]
        p2 = [pm[2, 1], pm[-1, 1]]
        self.do_polyprod_and_polydiv_test(p1, p2, pm)

    def test_polyprod_and_polydiv_2(self):
        pm = gf.gen_pow_matrix(76553)
        p1 = [pm[5, 1], pm[9, 1]]
        p2 = [pm[6, 1], pm[-2, 1]]
        self.do_polyprod_and_polydiv_test(p1, p2, pm)

    def test_polyadd(self):
        a = gf.polyadd([2, 3], [5, 10, 110])
        b = gf.polyadd([0, 2, 3], [5, 10, 110])
        assert_array_equal(a, b)
        assert_array_equal([5, 8, 109], b)

    def test_polyadd_2(self):
        a = gf.polyadd([1, 6, 12], [1, 7, 8])
        assert_array_equal([1, 4], a)

    def test_polyadd_3(self):
        a = gf.polyadd([1, 2], [1, 2])
        assert_array_equal(a, [0])

    def test_divide_zero(self):
        pm = gf.gen_pow_matrix(104155)
        self.assertRaises(BaseException, lambda: gf.divide(pm[-1, 1], 0, pm))

    def test_divide_zero_2(self):
        pm = gf.gen_pow_matrix(10187)
        for elem in pm[:, 1]:
            self.assertEqual(0, gf.divide(0, elem, pm))

    def test_divide_itself(self):
        pm = gf.gen_pow_matrix(54193)
        for elem in pm[:, 1]:
            self.assertEqual(1, gf.divide(elem, elem, pm))

    def test_divide_inverse(self):
        pm = gf.gen_pow_matrix(88479)
        for elem in pm[:, 1]:
            inverse = gf.divide(1, elem, pm)
            self.assertEqual(1, gf.prod(inverse, elem, pm))

    def test_divide_all(self):
        pm = gf.gen_pow_matrix(357)
        for elem1 in pm[:, 1]:
            for elem2 in pm[:, 1]:
                a = gf.divide(elem1, elem2, pm)
                self.assertEqual(elem1, gf.prod(a, elem2, pm))

    def test_euclid(self):
        pm = gf.gen_pow_matrix(37)
        p1 = np.array([2, 14, 22, 23, 8, 17, 1, 11, 26, 3])
        p2 = np.array([31, 23, 29, 31, 11, 9])
        max_deg = 3
        result = gf.euclid(p1, p2, pm, max_deg=max_deg)
        assert_array_equal(gf.polyadd(gf.polyprod(p1, result[1], pm), gf.polyprod(p2, result[2], pm)), result[0])

    def test_polydeg(self):
        self.assertEqual(0, gf.polydeg([0]))

    def test_polydeg_2(self):
        self.assertEqual(0, gf.polydeg([0, 0]))

    def test_polydeg_3(self):
        self.assertEqual(1, gf.polydeg([1, 1]))

    def test_linsolve(self):
        pm = gf.gen_pow_matrix(130207)
        A = [[pm[5, 1], pm[20, 1], pm[6, 1]],
             [0, 0, 0],
             [pm[-1, 1], pm[2, 1], pm[9, 1]]]
        self.assertTrue(gf.linsolve(A, [pm[1, 1], pm[5, 1], pm[-3, 1]], pm) is np.nan)

    def test_linsolve_2(self):
        pm = gf.gen_pow_matrix(130207)
        A = [[pm[5, 1], pm[20, 1], pm[6, 1]],
             [pm[5, 1], pm[20, 1], pm[6, 1]],
             [pm[-1, 1], pm[2, 1], pm[9, 1]]]
        self.assertTrue(gf.linsolve(A, [pm[1, 1], pm[5, 1], pm[-3, 1]], pm) is np.nan)

    def test_linsolve_3(self):
        pm = gf.gen_pow_matrix(108851)
        A = [[pm[5, 1], pm[20, 1], pm[6, 1]],
             [pm[8, 1], pm[1, 1], pm[6, 1]],
             [pm[-1, 1], pm[2, 1], pm[9, 1]]]
        assert_array_equal([3009, 23136, 63822], gf.linsolve(A, [pm[1, 1], pm[5, 1], pm[-3, 1]], pm))

    def test_linsolve_4(self):
        pm = gf.gen_pow_matrix(87341)
        A = [[pm[20, 1], pm[-20, 1], pm[10, 1], pm[8, 1]],
             [pm[198, 1], pm[30, 1], pm[89, 1], pm[-30, 1]],
             [pm[298, 1], pm[32, 1], pm[86, 1], pm[-24, 1]],
             [pm[5, 1], pm[-67, 1], pm[94, 1], pm[43, 1]]]
        b = [pm[67, 1], pm[-39, 1], pm[49, 1], pm[87, 1]]
        assert_array_equal([35048, 24262, 65502, 26384],
                           gf.linsolve(A, b, pm))

    def test_linsolve_5(self):
        pm = gf.gen_pow_matrix(19)
        A1 = np.array([[3, 7], [12, 1]])
        A2 = np.array([[3, 7], [12, 15]])
        b = np.array([8, 13])
        assert_array_equal(np.array([13, 14]), gf.linsolve(A1, b, pm))
        self.assertTrue(gf.linsolve(A2, b, pm) is np.nan)

    def test_linsolve_6(self):
        pm = gf.gen_pow_matrix(87341)
        A = [[pm[20, 1], pm[-20, 1], 0, pm[8, 1]],
             [pm[298, 1], pm[30, 1], 0, pm[-30, 1]],
             [pm[20, 1], pm[32, 1], 0, pm[-24, 1]],
             [pm[20, 1], pm[-67, 1], 0, pm[43, 1]]]
        self.assertTrue(gf.linsolve(A, [pm[67, 1], pm[-39, 1], pm[87, 1], pm[49, 1]], pm) is np.nan)

    def test_linsolve_7_with_zero(self):
        pm = gf.gen_pow_matrix(87341)
        A = [[0, pm[-20, 1], pm[10, 1], pm[8, 1]],
             [pm[198, 1], pm[30, 1], pm[89, 1], pm[-30, 1]],
             [pm[298, 1], pm[32, 1], pm[86, 1], pm[-24, 1]],
             [pm[5, 1], pm[-67, 1], pm[94, 1], pm[43, 1]]]
        b = [pm[67, 1], pm[-39, 1], pm[49, 1], pm[87, 1]]
        assert_array_equal([21320, 18899, 5953, 57137],
                           gf.linsolve(A, b, pm))

    def test_linsolve_random(self):
        while True:
            pm = gf.gen_pow_matrix(92127)
            pm_len = len(pm)
            n = 100
            A = [[pm[random.randint(0, pm_len - 1), 1] for _ in range(n)] for _ in range(n)]
            b = [pm[random.randint(0, pm_len - 1), 1] for _ in range(n)]
            solution = gf.linsolve(A, b, pm)
            if not (solution is np.nan):
                subst = [gf.sum(gf.prod(A[i], solution, pm)) for i in range(n)]
                assert_array_equal(subst, b)
                break

    def do_polyprod_and_polydiv_test(self, p1, p2, pm):
        div = gf.polydiv(p1, p2, pm)
        mult = gf.polyprod(div[0], p2, pm)
        assert_array_equal(p1, gf.add(mult, np.concatenate(
            [np.zeros(len(mult) - len(div[1])).astype(int), div[1]])))


if __name__ == '__main__':
    unittest.main()
