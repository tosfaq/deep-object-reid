from .common import ModelInterface
from torchreid.ops import Dropout
import timm

__ALL__ = ['tf_efficientnetv2_b0', 'tf_efficientnetv2_s_in21k', 'tf_efficientnet_lite1']

class EfficientNetTimm(ModelInterface):
    def __init__(self,
                model_name,
                pretrained=False,
                dropout_cls = None,
                pooling_type='avg',
                num_classes=1000,
                **kwargs):
        super().__init__(**kwargs)
        self.model = timm.create_model( model_name,
                                   pretrained=pretrained,
                                   num_classes=num_classes)

        self.num_features = self.model.num_features
        self.dropout = Dropout(**dropout_cls)
        self.pooling_type = pooling_type
        assert self.loss in ['softmax', 'asl'], "EfficientNetTimm supports only softmax or ASL losses"

    def forward(self, x, return_featuremaps=False, get_embeddings=False, gt_labels=None):
        assert not get_embeddings
        y = self.model.forward_features(x)
        if return_featuremaps:
            return y
        glob_features = self._glob_feature_vector(y, self.pooling_type, reduce_dims=False).view(x.shape[0], -1)
        self.dropout(glob_features)
        logits = self.model.classifier(glob_features)

        if not self.training and self.is_classification():
            return [logits]

        elif self.loss in ['softmax', 'am_softmax', 'asl']:
                out_data = [logits]
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        return tuple(out_data)


def tf_efficientnetv2_b0(**kwargs):
    """
    Constructs a tf_efficientnetv2_b0 model
    """
    net = EfficientNetTimm(model_name="tf_efficientnetv2_b0", **kwargs)
    return net


def tf_efficientnetv2_s_in21k(**kwargs):
    """
    Constructs a tf_efficientnetv2_s_in21k model
    """
    net = EfficientNetTimm(model_name="tf_efficientnetv2_s_in21k", **kwargs)
    return net


def tf_efficientnet_lite1(**kwargs):
    """
    Constructs a tf_efficientnet_lite1 model
    """
    net = EfficientNetTimm(model_name="tf_efficientnet_lite1", **kwargs)
    return net
