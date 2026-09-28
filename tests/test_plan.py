import copy
import json
from pathlib import Path
import unittest
from dysthe_experiments.__main__ import validate_plan

class Plan(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((Path(__file__).parents[1]/'configs/pilot.json').read_text())

    def test_planning_document(self):
        validate_plan(self.plan)

    def test_cannot_pretend_to_run_or_split_snapshots(self):
        for key,value in [('status','ready'),('split_unit','snapshot'),('locked_test',False),('methods',['unknown'])]:
            plan=copy.deepcopy(self.plan)
            plan[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(plan)
