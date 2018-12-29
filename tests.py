from common import *


class GFTests(NumpyTest):
    def setUp(self):
        self.basedir = os.path.realpath(os.path.dirname(__file__))
        self.pow_matrices = {}
        try:
            with np.load(os.path.join(self.basedir, 'pow_matrices.npz')) as pms:
                for primpoly, pm in pms.iteritems():
                    self.pow_matrices[int(primpoly)] = pm
        except FileNotFoundError:
            self.fail("pow_matrices.npz is missing, please download the full testing repo")

        self.arithmetic_tests = [
            # primpoly, a, b, (a*b, a/b, a+b, sum(a, axis=0), sum(a, axis=1), ..., sum(a, axis=-1))
            (19, A_(13), A_(1), (A_(13), A_(13), A_(12), A_(13))),
            (59, np.arange(10), 10 - np.arange(10),
             (
                 A_([0, 9, 16, 9, 24, 17, 24, 9, 16, 9]),
                 A_([0, 15, 19, 8, 23, 1, 28, 20, 4, 9]),
                 A_([10, 8, 10, 4, 2, 0, 2, 4, 10, 8]),
                 A_(1)
             )),
            (130207,
             A_([[40149, 51717, 13029, 10470],
                 [35757, 1468, 47617, 23044],
                 [46388, 60508, 8685, 1384]]),
             A_([[22129, 65478, 929, 12126],
                 [46610, 35400, 38929, 56596],
                 [30008, 13352, 57796, 25605]]),
             (
                 A_([[9535, 8756, 37217, 28517],
                     [61920, 12737, 22179, 40262],
                     [56217, 59163, 16552, 6823]]),
                 A_([[16049, 37452, 28777, 27218],
                     [53620, 60519, 35689, 5025],
                     [44202, 11489, 19327, 19722]]),
                 A_([[51876, 13763, 12612, 1976],
                     [15807, 36852, 8720, 34576],
                     [49164, 55412, 49193, 24941]]),
                 A_([41548, 9189, 43273, 30602]),
                 A_([19667, 28180, 32237])
             ))
        ]

        self.linsolve_tests = [
            # primpoly, A, b, linsolve(A, b)
            (130207, A_([[64, 23056, 128],
                         [0, 0, 0],
                         [1, 8, 1024]]),
             A_([4, 64, 33128]), np.nan),

            (130207, A_([[64, 23056, 128],
                         [64, 23056, 128],
                         [1, 8, 1024]]),
             A_([4, 64, 33128]), np.nan),

            (108851, A_([[64, 1949, 128],
                         [512, 4, 128],
                         [1, 8, 1024]]),
             A_([4, 64, 48853]), A_([3009, 23136, 63822])),

            (87341, A_([[3272, 59574, 2048, 512],
                        [15319, 54747, 28268, 58909],
                        [59446, 43035, 42843, 56307],
                        [64, 11873, 39430, 27645]]),
             A_([21004, 40721, 20556, 7067]), A_([35048, 24262, 65502, 26384])),

            (19, A_([[3, 7], [12, 1]]), A_([8, 13]), A_([13, 14])),
            (19, A_([[3, 7], [12, 15]]), A_([8, 13]), np.nan),

            (87341, A_([[3272, 59574, 0, 512],
                        [59446, 54747, 0, 58909],
                        [3272, 43035, 0, 56307],
                        [3272, 11873, 0, 27645]]),
             A_([21004, 40721, 7067, 20556]), np.nan),

            (87341, A_([[0, 59574, 2048, 512],
                        [15319, 54747, 28268, 58909],
                        [59446, 43035, 42843, 56307],
                        [64, 11873, 39430, 27645]]),
             A_([21004, 40721, 20556, 7067]), A_([21320, 18899, 5953, 57137])),

            (87341, A_([[1, 59574, 2048, 512],
                        [1, 59574, 28268, 58909],
                        [59446, 43035, 42843, 56307],
                        [64, 11873, 39430, 27645]]),
             A_([21004, 40721, 20556, 7067]), A_([49980, 29479, 12587, 62413]))
        ]

        self.minpoly_tests = [
            # primpoly, x, minpoly, x_conjugates
            (11, A_([2]), A_([1, 0, 1, 1]), A_([2, 4, 6])),
            (19, A_([0, 2]), A_([1, 0, 0, 1, 1, 0]), A_([0, 2, 3, 4, 5]))
        ]

        self.polyval_tests = [
            # primpoly, p, x, polyval(p, x)
            (19, A_([1, 1, 0, 0, 1]), A_([5]), A_([15])),
            (59, np.arange(6)[::-1],
             A_([0, 25, 1, 19, 27, 9, 13, 11, 31, 14, 2, 6, 4, 17, 5,
                 10, 20, 30, 3, 15, 7, 28, 23, 22, 21, 16, 18, 12, 26]),
             A_([0, 28, 1, 26, 13, 13, 21, 6, 27, 16, 30, 5, 20, 3, 14,
                 28, 30, 19, 2, 17, 2, 16, 27, 21, 13, 11, 15, 9, 30])),
            (130207, np.arange(999)[::-1],
             A_([0, 25, 1, 19, 27, 9, 13, 11, 31, 14, 2, 6, 4, 17, 5,
                 10, 20, 30, 3, 15, 7, 28, 23, 22, 21, 16, 18, 12, 26]),
             A_([0, 17099, 999, 30857, 18242, 61694, 29395, 17685, 60242, 2445, 18703, 56015, 33933, 38500, 3990,
                 61825, 51582, 54841, 26026, 55158, 13047, 49000, 57703, 612, 24390, 27617, 15813, 36465, 54278]))
        ]

    def test_00_gen_pow_matrix(self):
        """
        Test power matrix generation.
        Note: this requires the pow_matrices.npz file to be present
        """
        self.assertTrue(len(self.pow_matrices), msg='pow_matrices.npz is empty')
        for primpoly in self.pow_matrices.keys():
            with self.subTest(primpoly=primpoly):
                self.assertNdarrayEqual(gf.gen_pow_matrix(primpoly), self.pow_matrices[primpoly])

    def test_01_arithmetic(self):
        """
        Test the 4 basic operations - prod, divide, add and sum. (see self.arithmetic_tests)
        """
        for idx, (primpoly, a, b, op_res) in enumerate(self.arithmetic_tests):
            pm = self.pow_matrices[primpoly]
            add_res = op_res[2]
            # Addition
            with self.subTest(idx=idx, op='add'):
                self.assertNdarrayEqual(gf.add(a, b), add_res)
            # Multiplication and Division
            for op, opname, res in zip((gf.prod, gf.divide), ('prod', 'divide'), op_res[:2]):
                with self.subTest(idx=idx, primpoly=primpoly, op=opname):
                    self.assertNdarrayEqual(op(a, b, pm), res)
            sums = op_res[3:]
            # Summation along all possible axes
            for ax, sum_res in enumerate(sums):
                with self.subTest(idx=idx, op='sum', axis=ax):
                    self.assertNdarrayEqual(gf.sum(a, axis=ax), sum_res)
            sum_last = sums[-1]
            with self.subTest(idx=idx, op='sum', axis=-1):
                self.assertNdarrayEqual(gf.sum(a, axis=-1), sum_last)
            # Division by zero and division of zero
            with self.subTest(idx=idx, primpoly=primpoly, op='divide'):
                with self.assertRaises(BaseException):
                    gf.divide(a, A_([0]), pm)
                self.assertNdarrayEqual(gf.divide(A_([0]), b, pm), np.broadcast_to(0, b.shape))

    def test_02_prod_divide(self):
        """
        Test that (a/b)*b == a for random elements in various fields.
        """
        n = 1000
        for primpoly in [19, 59, 357, 54193, 88479, 104155]:
            pm = self.pow_matrices[primpoly]
            e1 = np.random.permutation(pm[:, 1])[:n, np.newaxis]
            e2 = np.random.permutation(pm[:, 1])[:n]
            e1, e2 = np.broadcast_arrays(e1, e2)
            with self.subTest(primpoly=primpoly):
                self.assertNdarrayEqual(gf.prod(gf.divide(e1, e2, pm), e2, pm), e1)

    def test_03_linsolve(self):
        """
        Test linsolve with some known solutions. (see self.linsolve_tests)
        """
        for idx, (primpoly, A, b, result) in enumerate(self.linsolve_tests):
            with self.subTest(idx=idx, primpoly=primpoly, result=result):
                pm = self.pow_matrices[primpoly]
                check = gf.linsolve(A, b, pm)
                if result is np.nan:
                    self.assertIs(check, np.nan)
                else:
                    self.assertNdarrayEqual(gf.linsolve(A, b, pm), result)

    def test_04_linsolve_random(self):
        """
        Test that A @ linsolve(A, b) == b for random elements in various fields.
        """
        num_tests = 100
        n = 100

        for idx in range(num_tests):
            primpoly = np.random.choice(list(self.pow_matrices.keys()))
            pm = self.pow_matrices[primpoly]
            pm_len = len(pm)
            with self.subTest(idx=idx, primpoly=primpoly):
                solution = np.nan
                while solution is np.nan:
                    A = np.take(pm[:, 1], np.random.randint(pm_len - 1, size=(n, n)))
                    b = np.take(pm[:, 1], np.random.randint(pm_len - 1, size=n))
                    solution = gf.linsolve(A, b, pm)
                solution = np.broadcast_to(solution, A.shape)
                self.assertNdarrayEqual(gf.sum(gf.prod(A, solution, pm), axis=-1), b)

    def test_05_minpoly(self):
        """
        Tests minpoly with some known solutions. (see self.minpoly_tests)
        """
        for idx, (primpoly, xs, res_mp, res_xc) in enumerate(self.minpoly_tests):
            with self.subTest(idx=idx, primpoly=primpoly):
                pm = self.pow_matrices[primpoly]
                mp, xc = gf.minpoly(xs, pm)
                self.assertNdarrayEqual(mp, res_mp)
                self.assertNdarrayEqual(xc, res_xc)

    def test_06_polyval(self):
        """
        Tests polyval with some known values. (see self.polyval_tests)
        """
        for idx, (primpoly, p, x, res) in enumerate(self.polyval_tests):
            with self.subTest(idx=idx, primpoly=primpoly):
                pm = self.pow_matrices[primpoly]
                self.assertNdarrayEqual(gf.polyval(p, x, pm), res)

    # TODO:
    def test_07_polyprod(self):
        self.assertNdarrayEqual(A_([1, 0b111, 0b111]),
                                gf.polyprod(A_([1, 0b11]), A_([1, 0b100]), self.pow_matrices[0b1011]))

        # n1
        pm = self.pow_matrices[19]
        p1 = A_([pm[5, 1], pm[-1, 1]])
        zero = A_([0])
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), A_([0]))

        # n2
        pm = self.pow_matrices[19]
        p1 = A_([pm[-3, 1], pm[-1, 1]])
        zero = A_([0, 0])
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), A_([0]))

        # n3
        pm = self.pow_matrices[19]
        p1 = A_([0, pm[-3, 1], pm[-1, 1]])
        zero = A_([0])
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), A_([0]))

        # n4
        pm = self.pow_matrices[19]
        p1 = A_([0, pm[-3, 1], pm[-1, 1]])
        zero = A_([0, 0])
        self.assertNdarrayEqual(gf.polyprod(p1, zero, pm), A_([0]))

    # TODO:
    def test_08_polydiv(self):
        div = gf.polydiv(A_([0b10, 0b1]), A_([0b1]), self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], A_([0b10, 0b1]))
        self.assertNdarrayEqual(div[1], A_([0b0]))

        # n1
        div = gf.polydiv(A_([0, 0b10, 0b1]), A_([0b1]), self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], A_([0b10, 0b1]))
        self.assertNdarrayEqual(div[1], A_([0b0]))

        # n2
        div = gf.polydiv(A_([0b10, 0b1]), A_([0, 0b1]), self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], A_([0b10, 0b1]))
        self.assertNdarrayEqual(div[1], A_([0b0]))

        # n3
        div = gf.polydiv(A_([0, 0b10, 0b1]), A_([0, 0b1]), self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], A_([0b10, 0b1]))
        self.assertNdarrayEqual(div[1], A_([0b0]))

        # 2
        div = gf.polydiv(A_([0b10, 0b1]), A_([0b10]), self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], A_([0b1, 0b101]))
        self.assertNdarrayEqual(div[1], A_([0b0]))

        # 3
        div = gf.polydiv(A_([0b10, 0b1]), A_([0b10, 0b0]), self.pow_matrices[0b1011])
        self.assertNdarrayEqual(div[0], A_([0b1]))
        self.assertNdarrayEqual(div[1], A_([0b1]))

        # div by zero
        pm = self.pow_matrices[5391]
        for elem in pm[:, 1]:
            self.assertRaises(BaseException, lambda: gf.polydiv(A_([elem]), A_([0]), pm))

    # TODO:
    def test_09_polyprod_polydiv(self):
        pm = self.pow_matrices[108439]
        p1 = A_([pm[5, 1], pm[3, 1]])
        p2 = A_([pm[2, 1], pm[-1, 1]])
        self._polyprod_polydiv(p1, p2, pm)

        # 2
        pm = self.pow_matrices[76553]
        p1 = A_([pm[5, 1], pm[9, 1]])
        p2 = A_([pm[6, 1], pm[-2, 1]])
        self._polyprod_polydiv(p1, p2, pm)

    # TODO:
    def test_10_euclid(self):
        pm = self.pow_matrices[37]
        p1 = A_([2, 14, 22, 23, 8, 17, 1, 11, 26, 3])
        p2 = A_([31, 23, 29, 31, 11, 9])
        max_deg = 3
        result = gf.euclid(p1, p2, pm, max_deg=max_deg)
        self.assertNdarrayEqual(gf.polyadd(gf.polyprod(p1, result[1], pm), gf.polyprod(p2, result[2], pm)), result[0])

    # TODO: (remove)
    def _polyprod_polydiv(self, p1, p2, pm):
        div = gf.polydiv(p1, p2, pm)
        mult = gf.polyprod(div[0], p2, pm)
        self.assertNdarrayEqual(p1, gf.add(mult, np.concatenate(
            [np.zeros(len(mult) - len(div[1])).astype(int), div[1]])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
