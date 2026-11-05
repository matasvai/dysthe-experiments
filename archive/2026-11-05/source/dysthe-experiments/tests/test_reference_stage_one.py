import unittest

from dysthe_experiments.reference_stage_one import configuration
from dysthe_experiments.solitary import validate


class StageOnePlanTests(unittest.TestCase):
    def test_fixed_spacing_domain_plan_and_bounded_work(self):
        config = validate(configuration('/tmp/profile.npz'))
        domain = [c for c in config['cases'] if c['id'].startswith('domain-')]
        self.assertEqual([c['length']/c['n'] for c in domain], [160/4096]*2)
        self.assertTrue(all(c['xi_end'] == 20 and c['epsilon'] == .025 for c in domain))
        self.assertEqual(sum(len(c['perturbations']) for c in config['cases']), 8)

    def test_explicit_largest_domain_and_benchmark_are_exclusive(self):
        config = validate(configuration('/tmp/profile.npz', extended_domain=True))
        self.assertEqual(len(config['cases']), 1)
        self.assertEqual(config['cases'][0]['n'], 32768)
        self.assertEqual(config['cases'][0]['length'], 1280.)
        with self.assertRaises(ValueError):
            configuration('/tmp/profile.npz', benchmark=True, extended_domain=True)


if __name__ == '__main__':
    unittest.main()
