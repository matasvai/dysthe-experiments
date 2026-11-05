import copy
import json
from pathlib import Path
import unittest
from dysthe_core import validate_case, validate_splits

class Contracts(unittest.TestCase):
    def setUp(self):
        self.case = json.loads((Path(__file__).parents[1] / 'examples/case.json').read_text())

    def test_example_and_refinements(self):
        validate_case(self.case)
        validate_splits([dict(case_id='coarse',initial_condition_id='a',split='train'),
                         dict(case_id='fine',initial_condition_id='a',split='train'),
                         dict(case_id='other',initial_condition_id='b',split='test')])

    def test_refined_copy_cannot_leak_into_test(self):
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_splits([dict(case_id='coarse',initial_condition_id='a',split='train'),
                             dict(case_id='fine',initial_condition_id='a',split='test')])

    def test_invalid_physics_layout_and_time(self):
        for key, value in [('model_id','vlasov-poisson'), ('propagation_positions',[0,0]),
                           ('propagation_positions',[0,float('nan')]), ('axis_order',['t','v','x']),
                           ('shape',[True,16,16]), ('lengths',[20,-1,20])]:
            case = copy.deepcopy(self.case)
            case[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_case(case)

    def test_duplicate_case_rejected(self):
        case = dict(case_id='same',initial_condition_id='a',split='train')
        with self.assertRaises(ValueError):
            validate_splits([case, case])

    def test_old_optical_contract_cannot_be_relabelled(self):
        changes = [('schema_version', 1), ('model_id', 'optical-dysthe-127-v1'),
                   ('physical_system', 'plasma'), ('shape', [16,16,16]),
                   ('physics', {'epsilon1':0.01,'epsilon2':0.01}),
                   ('physics', {'epsilon':0.05,'epsilon1':0.01}),
                   ('times', [0,0.05])]
        for key, value in changes:
            case = copy.deepcopy(self.case)
            case[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_case(case)

    def test_steepness_and_nls_control(self):
        for epsilon in [0, 0.05]:
            self.case['physics'] = {'epsilon':epsilon}
            validate_case(self.case)
        for epsilon in [-0.1, float('nan'), float('inf'), True, '0.05']:
            self.case['physics'] = {'epsilon':epsilon}
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                validate_case(self.case)
