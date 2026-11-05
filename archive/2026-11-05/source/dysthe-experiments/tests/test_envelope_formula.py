"""Analytic agreement and rejection boundaries for the small formula grammar."""
import json
import unittest

import numpy as np

from dysthe_experiments.envelope_formula import (
    FormulaError, FORMULA_DEFAULT, MAX_NODES, PARSER_VERSION, SYMBOL_DEFAULTS,
    compile_formula, validate_formula,
)


class EnvelopeFormulaTests(unittest.TestCase):
    def setUp(self):
        self.tau = np.linspace(-12., 12., 513)

    def test_default_matches_existing_gaussian_prescription_with_both_channels(self):
        # This is the simulator's existing Gaussian prescription, including chirp.
        symbols = dict(SYMBOL_DEFAULTS, A=.8, w=2.2, k=.25, c=.01)
        actual = compile_formula(FORMULA_DEFAULT, symbols).evaluate(self.tau)
        expected = (symbols['A']*np.exp(-self.tau**2/(2*symbols['w']**2))
                    * np.exp(1j*(symbols['k']*self.tau+symbols['c']*self.tau**2)))
        self.assertEqual(actual.dtype, np.complex128)
        np.testing.assert_allclose(actual, expected, rtol=4e-15, atol=3e-16)
        self.assertGreater(np.linalg.norm(actual.real), 1.)
        self.assertGreater(np.linalg.norm(actual.imag), .1)

    def test_sampled_default_matches_existing_simulator_initializer(self):
        from dysthe_core.reference import spatial
        from dysthe_experiments.simulator import initial_field

        model = spatial.SpatialDysthe(512, 40., .025)
        for chirp in (0., .01):
            with self.subTest(chirp=chirp):
                compiled = compile_formula(FORMULA_DEFAULT, dict(SYMBOL_DEFAULTS, c=chirp))
                field, reference, source = initial_field('gaussian', model, dict(chirp=chirp))
                self.assertIsNone(reference)
                self.assertLess(source['projection_relative_l2'], 1e-12)
                np.testing.assert_allclose(compiled.evaluate(model.tau), field,
                                           rtol=0, atol=1e-13)

    def test_sech_modulated_and_displaced_packet(self):
        formula = r'A \sech(\frac{\tau-b}{w}) \exp(i(k\tau+c\tau^2))'
        symbols = dict(A=1.1, w=1.8, b=.4, k=-.3, c=.02)
        actual = compile_formula(formula, symbols).evaluate(self.tau)
        expected = (symbols['A']/np.cosh((self.tau-symbols['b'])/symbols['w'])
                    * np.exp(1j*(symbols['k']*self.tau+symbols['c']*self.tau**2)))
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-16)

    def test_plain_text_latex_delimiters_and_trigonometric_functions(self):
        formula = r'A \cdot \left(1 + b \cos(k \tau)\right) \times exp(i sin(tau) + i pi/4)'
        symbols = dict(A=.4, b=.1, k=.2)
        actual = compile_formula(formula, symbols).evaluate(self.tau)
        expected = .4*(1+.1*np.cos(.2*self.tau))*np.exp(1j*np.sin(self.tau)+1j*np.pi/4)
        np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=3e-16)
        adjacent = compile_formula('2 A w tau', {'A':.2, 'w':3}).evaluate(self.tau)
        np.testing.assert_allclose(adjacent, 1.2*self.tau, atol=4e-15)

    def test_precedence_integer_powers_and_left_associative_division(self):
        axis = np.array([2., 3.])
        for formula, expected in [
            ('-tau^2', -axis**2), ('(-tau)^2', axis**2),
            ('tau^{-2}', axis**-2), ('tau^(+2)', axis**2),
            ('tau^0', np.ones(2)), ('8/2 tau', 4*axis), ('1e-3 tau', .001*axis),
        ]:
            with self.subTest(formula=formula):
                np.testing.assert_allclose(compile_formula(formula, {}).evaluate(axis), expected)

    def test_scalar_broadcast_and_provenance_are_independent_json_safe_copies(self):
        declared = dict(A=.7)
        compiled = validate_formula('A+i/2', declared)
        declared['A'] = 9
        actual = compiled.evaluate(self.tau)
        self.assertEqual(actual.shape, self.tau.shape)
        np.testing.assert_array_equal(actual, np.full(self.tau.shape, .7+.5j))
        actual[0] = 8
        self.assertEqual(compiled.evaluate(self.tau)[0], .7+.5j)
        source = compiled.provenance()
        self.assertEqual(source['parser_version'], PARSER_VERSION)
        self.assertEqual(source['formula'], 'A+i/2')
        self.assertEqual(source['symbols'], {'A':.7})
        self.assertIn('normalized_expression', source)
        self.assertLessEqual(source['expression_nodes'], MAX_NODES)
        json.dumps(source, allow_nan=False)
        source['symbols']['A'] = 6
        self.assertEqual(compiled.provenance()['symbols'], {'A':.7})

    def test_zeros_and_underflowed_tails_are_finite_parser_outputs(self):
        # Zero action is rejected by simulation integration, not the parser.
        np.testing.assert_array_equal(compile_formula('0', {}).evaluate(self.tau), np.zeros(self.tau.shape))
        out = compile_formula('exp(-tau^2)', {}).evaluate(np.array([-100., 0., 100.]))
        np.testing.assert_array_equal(out, [0., 1., 0.])

    def test_rejects_execution_strings_and_unsupported_notation(self):
        formulas = [
            '__import__("os").system("touch /tmp/unsafe")', 'open(tau)',
            'tau.__class__', 'tau[0]', 'tau;1', 'lambda x:x', 'sum(tau)',
            r'\int tau', r'\sum_{k=1}^{10} tau', r'\sqrt(tau)',
            r'\operatorname{sech}(tau)', r'\unknown(tau)', 'log(tau)',
            'u(0,tau)=exp(-tau^2)', '$exp(-tau^2)$', '\\', 'π tau',
            r'\left[tau\right]', r'\left tau', 'A(tau)',
        ]
        for formula in formulas:
            with self.subTest(formula=formula), self.assertRaises(FormulaError):
                # A(tau) is multiplication if A is declared; it is undeclared here.
                compile_formula(formula, {})

    def test_rejects_malformed_grammar_and_powers(self):
        formulas = ['', ' ', None, [], 1, '(', 'tau)', '()', 'tau+',
                    'tau**2', 'tau^', 'tau^tau', 'tau^2.0', 'tau^.5',
                    'tau^13', 'tau^{-13}', 'tau^999999', 'tau^2^2',
                    'tau^(2+1)', r'\frac(1)(2)', r'\frac{1}',
                    r'\frac{}{2}', 'sin tau', 'exp()', '(tau}', 'tau,1']
        for formula in formulas:
            with self.subTest(formula=formula), self.assertRaises(FormulaError):
                compile_formula(formula, {})

    def test_rejects_bad_symbol_declarations(self):
        symbols = [None, [], 4, {'A':True}, {'A':'1'}, {'A':1j},
                   {'A':[1]}, {'A':np.array(1.)}, {'A':np.float64(1)},
                   {'A':float('nan')}, {'A':float('inf')}, {'A':-float('inf')},
                   {'A':1e7}, {'A':10**1000}, {'i':1}, {'tau':1},
                   {'width':1}, {'_':1}, {'é':1}, {1:1},
                   {letter:1 for letter in 'abcdefghj'}]
        for declared in symbols:
            with self.subTest(symbols=declared), self.assertRaises(FormulaError):
                compile_formula('1', declared)
        with self.assertRaisesRegex(FormulaError, 'Undeclared symbol'):
            compile_formula('a tau', {'b':1})

    def test_rejects_formula_complexity_and_numeric_literal_work(self):
        balanced = '1'
        for _ in range(8):
            balanced = '('+balanced+'+'+balanced+')'
        formulas = ['1 '*600, '('*40+'tau'+')'*40, '-'*40+'tau',
                    '+'.join(['1']*40), balanced, '9'*33,
                    '1e300', '1e999999999', '1000001']
        for formula in formulas:
            with self.subTest(formula=formula[:50]), self.assertRaises(FormulaError):
                compile_formula(formula, {})

    def test_rejects_division_singularities_overflow_and_excessive_intermediates(self):
        formulas = ['1/0', '1/(tau-tau)', 'tau^{-1}', 'exp(1000)',
                    'exp(exp(20))', '1000000^3/1000000^3',
                    'cos(1000 i)', 'sin(1000 i)', 'sech(1000)', '1/1e-300']
        for formula in formulas:
            with self.subTest(formula=formula), self.assertRaises(FormulaError):
                compile_formula(formula, {}).evaluate(np.array([0., 1.]))

    def test_rejects_invalid_axes_and_retains_sample_bound(self):
        formula = compile_formula('1', {})
        axes = [np.zeros((2, 2)), np.array(1.), np.array([]), np.zeros(16385),
                np.array([np.nan]), np.array([np.inf]), np.array([1j]),
                np.array([True]), np.array(['1']), np.array([1], dtype=object),
                np.array([-2**63], dtype=np.int64), np.array([1000001.])]
        for axis in axes:
            with self.subTest(shape=axis.shape, dtype=axis.dtype), self.assertRaises(FormulaError):
                formula.evaluate(axis)
        self.assertEqual(formula.evaluate(np.zeros(16384)).size, 16384)


if __name__ == '__main__':
    unittest.main()
