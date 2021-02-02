from .fpn import FPN
from .gmp import GeneralizedMeanPooling
from .norm import LocalContrastNormalization
from .utils import EvalModeSetter
from .gumbel import GumbelSigmoid, GumbelSoftmax, gumbel_sigmoid
from .dropout import Dropout
from .non_local import NonLocalModule
from .data_parallel import DataParallel
from .nonlinearities import HSwish, HSigmoid
from .self_challenging import RSC, rsc
