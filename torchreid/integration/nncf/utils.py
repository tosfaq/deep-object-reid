from torchreid.integration.nncf.compression import NNCFMetaInfo
from nncf.config.utils import is_accuracy_aware_training


def get_no_nncf_trace_context_manager():
    try:
        from nncf.torch.dynamic_graph.context import no_nncf_trace
        return no_nncf_trace
    except ImportError:
        @contextmanager
        def nullcontext():
            """
            Context which does nothing
            """
            yield

        return nullcontext


def is_checkpoint_nncf(filename: str) -> bool:
    nncf_metainfo = NNCFMetaInfo.get_from_checkpoint(filename)
    if nncf_metainfo is None:
        return False
    return nncf_metainfo.compression_enabled
