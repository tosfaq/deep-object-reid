import logging
import os
import re

from subprocess import run, CalledProcessError

if __name__ == '__main__':
    ignored_patterns = [
        'setup.py',
        'build',
        'tools',
        'projects',
        '.history',
        'torchreid/models',
        'torchreid/data',
        'torchreid/engine',
        'torchreid/metrics',
        'torchreid/optim',
        'torchreid/utils',
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

    try:
        run(['pylint'] + to_pylint, check=True)
    except CalledProcessError:
        logging.error('pylint check failed.')
