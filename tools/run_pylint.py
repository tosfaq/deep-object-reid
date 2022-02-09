import os
import re

from subprocess import run  # nosec

if __name__ == '__main__':
    ignored_patterns = [
        'setup.py',
        'tools/auxiliary/',
        '.history',
        'torchreid/models',
        'torchreid/optim/radam.py',
        'torchreid/optim/sam.py',
        'tests/conftest.py',
        'tests/config.py',
        'torchreid/integration/sc/model_wrappers/classification.py',
        'tests/test_ote_training.py'
    ]

    to_pylint = []
    wd = os.path.abspath('.')
    for root, dirnames, filenames in os.walk(wd):
        for filename in filenames:
            if filename.endswith('.py'):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, wd)
                if all(not re.match(pattern, rel_path) for pattern in ignored_patterns):
                    to_pylint.append(rel_path)

    run(['pylint'] + to_pylint, check=True)
