import e2e.fixtures

from e2e.conftest_utils import * # noqa
from e2e.utils import get_plugins_from_packages
pytest_plugins = get_plugins_from_packages([e2e])


def pytest_addoption(parser):
    parser.addoption('--dataset-definitions', action='store', default=None,
                     help='Path to the dataset_definitions.yml file for tests that require datasets.')
    parser.addoption('--test-usecase', action='store', default=None,
                     help='Optional. If the parameter is set, it filters test_ote_training tests by usecase field.')
    parser.addoption('--expected-metrics-file', action='store', default=None,
                     help='Optional. If the parameter is set, it points the YAML file with expected test metrics.')
