from .data_parallel import DataParallel
from .dropout import Dropout
from .fpn import FPN
from .gmp import GeneralizedMeanPooling
from .gumbel import GumbelSigmoid, GumbelSoftmax, gumbel_sigmoid
from .non_local import NonLocalModule
from .nonlinearities import HSigmoid, HSwish
from .norm import LocalContrastNormalization
from .self_challenging import RSC, rsc
from .utils import EvalModeSetter
