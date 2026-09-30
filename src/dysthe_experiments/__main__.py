import argparse
import json
from pathlib import Path
from dysthe_core import MODEL_ID, require_water_wave_model
from dysthe_learning import METHODS

def validate_plan(plan):
    require_water_wave_model(plan)
    if plan.get('schema_version') != 2 or plan.get('axis_order') != ['xi', 'tau']:
        raise ValueError('Unsupported campaign schema/model')
    if plan.get('status') != 'planning_only':
        raise ValueError('Execution is not implemented; only planning_only is supported')
    methods = plan.get('methods', [])
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('Choose unique registered methods')
    if plan.get('split_unit') != 'initial_condition_id' or not plan.get('locked_test'):
        raise ValueError('Entire initial-condition groups and a locked test set are required')
    if plan.get('early_stopping', {}).get('enabled') is not True:
        raise ValueError('The campaign must plan for early stopping')

def main():
    parser = argparse.ArgumentParser(description='Validate a campaign plan; no jobs are launched.')
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    try:
        validate_plan(json.loads(args.config.read_text()))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f'Invalid plan: {error}\n')
    print('Plan structure valid. Execution is not implemented; thresholds and case bounds remain to be set.')

if __name__ == '__main__':
    main()
