import unittest
from dysthe_learning import METHODS, require_implemented

class Registry(unittest.TestCase):
    def test_scaffold_cannot_start_training(self):
        for name in METHODS:
            with self.subTest(name=name), self.assertRaises(NotImplementedError):
                require_implemented(name)

    def test_gp_has_a_distinct_prediction_target(self):
        self.assertEqual(METHODS['active-gp']['target'], 'observable')
        self.assertTrue(all(v['target'] == 'field' for k,v in METHODS.items() if k != 'active-gp'))
