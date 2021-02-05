"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file', 'load_model', 'download_model', 'calc_num_params']

import hashlib
import logging
import os
import zipfile

_model_sha1 = {name: (error, checksum, repo_release_tag, caption, paper, ds, img_size, scale, batch, rem) for
               name, error, checksum, repo_release_tag, caption, paper, ds, img_size, scale, batch, rem in [
    ('mobilenetv2_wd4', '2451', '05e1e3a286b27c17ea11928783c4cd48b1e7a9b2', 'v0.0.137', 'MobileNetV2 x0.25', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2_wd2', '1493', 'b82d79f6730eac625e6b55b0618bff8f7a1ed86d', 'v0.0.170', 'MobileNetV2 x0.5', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2_w3d4', '1082', '8656de5a8d90b29779c35c5ce521267c841fd717', 'v0.0.230', 'MobileNetV2 x0.75', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2_w1', '0887', '13a021bca5b679b76156829743f7182da42e8bb6', 'v0.0.213', 'MobileNetV2 x1.0', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_wd4', '2530', 'f0e9b1208ebe0a83181b47c58b16f7bb7593674a', 'v0.0.453', 'MobileNetV2b x0.25', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_wd2', '1498', '489399099f540d61b00f99e3ed07df0eddb6325e', 'v0.0.453', 'MobileNetV2b x0.5', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_w3d4', '1178', '0cba52d7f9ce6bbef34049339dcf0f5f94dd57e2', 'v0.0.453', 'MobileNetV2b x0.75', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv2b_w1', '0975', 'c123d420973d53880164a67288704c1ac1153486', 'v0.0.453', 'MobileNetV2b x1.0', '1801.04381', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('mobilenetv3_large_w1', '0779', '38e392f58bdf99b2832b26341bc9704ac63a3672', 'v0.0.411', 'MobileNetV3 L/224/1.0', '1905.02244', 'in1k', 224, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('inceptionv3', '0565', 'cf4061800bc1dc3b090920fc9536d8ccc15bb86e', 'v0.0.92', 'InceptionV3', '1512.00567', 'in1k', 299, 0.875, 200, '[dmlc/gluon-cv]'),  # noqa
    ('inceptionv4', '0529', '5cb7b4e4b8f62d6b4346855d696b06b426b44f3d', 'v0.0.105', 'InceptionV4', '1602.07261', 'in1k', 299, 0.875, 200, '[Cadene/pretrained...pytorch]'),  # noqa
    ('inceptionresnetv2', '0490', '1d1b4d184e6d41091c5ac3321d99fa554b498dbe', 'v0.0.107', 'InceptionResNetV2', '1602.07261', 'in1k', 299, 0.875, 200, '[Cadene/pretrained...pytorch]'), # noqa
    ('efficientnet_b0', '0752', '0e3861300b8f1d1d0fb1bd15f0e06bba1ad6309b', 'v0.0.364', 'EfficientNet-B0', '1905.11946', 'in1k', 224, 0.875, 200, ''),  # noqa
    ('efficientnet_b1', '0638', 'ac77bcd722dc4f3edfa24b9fb7b8f9cece3d85ab', 'v0.0.376', 'EfficientNet-B1', '1905.11946', 'in1k', 240, 0.882, 200, ''),  # noqa
    ('efficientnet_b0b', '0702', 'ecf61b9b50666a6b444a9d789a5ff1087c65d0d8', 'v0.0.403', 'EfficientNet-B0b', '1905.11946', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b1b', '0594', '614e81663902850a738fa6c862fe406ecf205f73', 'v0.0.403', 'EfficientNet-B1b', '1905.11946', 'in1k', 240, 0.882, 200, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b2b', '0527', '531f10e6898778b7c3a82c2c149f8b3e6393a892', 'v0.0.403', 'EfficientNet-B2b', '1905.11946', 'in1k', 260, 0.890, 100, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b3b', '0445', '3c5fbba8c86121d4bc3bbc169804f24dd4c3d1f6', 'v0.0.403', 'EfficientNet-B3b', '1905.11946', 'in1k', 300, 0.904, 90, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b4b', '0389', '6305bfe688b261f0d4fef6829f520d5c98c46301', 'v0.0.403', 'EfficientNet-B4b', '1905.11946', 'in1k', 380, 0.922, 80, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b5b', '0337', 'e1c2ffcf710cbd3c53b9c08723282a370906731c', 'v0.0.403', 'EfficientNet-B5b', '1905.11946', 'in1k', 456, 0.934, 70, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b6b', '0323', 'e5c1d7c35fcff5fac07921a7696f7c04aba84012', 'v0.0.403', 'EfficientNet-B6b', '1905.11946', 'in1k', 528, 0.942, 60, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b7b', '0322', 'b9c5965a1e2572aaa772e20e8a2e3af7b4bee9a6', 'v0.0.403', 'EfficientNet-B7b', '1905.11946', 'in1k', 600, 0.949, 50, '[rwightman/pyt...models]'),  # noqa
    ('efficientnet_b0c', '0675', '21778c6e3b5a1b9b08b60c3e69401ce7e12bead4', 'v0.0.433', 'EfficientNet-B0с', '1905.11946', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b1c', '0569', '239ed6a412530f60f810b29807da70c8ca63d8cc', 'v0.0.433', 'EfficientNet-B1с', '1905.11946', 'in1k', 240, 0.882, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b2c', '0503', 'be48d3d79f25a13a807b137d8a7ced41e8aab2bf', 'v0.0.433', 'EfficientNet-B2с', '1905.11946', 'in1k', 260, 0.890, 100, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b3c', '0442', 'ea7080aba3fc20ac25c3c925bfadf1e8c1e7df4d', 'v0.0.433', 'EfficientNet-B3с', '1905.11946', 'in1k', 300, 0.904, 90, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b4c', '0369', '5954cc05cfba3b0c8ee488b4488354fc0cef6623', 'v0.0.433', 'EfficientNet-B4с', '1905.11946', 'in1k', 380, 0.922, 80, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b5c', '0310', '589fefc6de5d93b54698b5b03f1e05637f9d0cb6', 'v0.0.433', 'EfficientNet-B5с', '1905.11946', 'in1k', 456, 0.934, 70, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b6c', '0296', '546e61da82bec69e3db5870b8df977e4615f7b32', 'v0.0.433', 'EfficientNet-B6с', '1905.11946', 'in1k', 528, 0.942, 60, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b7c', '0288', '13d683f2ca56c1007acd9ad0be450f45efeec828', 'v0.0.433', 'EfficientNet-B7с', '1905.11946', 'in1k', 600, 0.949, 50, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_b8c', '0276', 'a9973d66d599c4e83029577842c039a20799f2c9', 'v0.0.433', 'EfficientNet-B8с', '1905.11946', 'in1k', 672, 0.954, 50, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_edge_small_b', '0640', 'e27c3444406ebddd86824e41a924c0b8188c4067', 'v0.0.434', 'EfficientNet-Edge-Small-b', '1905.11946', 'in1k', 224, 0.875, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_edge_medium_b', '0563', '99fa34c7044281e521fb7cf4267763a5b03b7f1c', 'v0.0.434', 'EfficientNet-Edge-Medium-b', '1905.11946', 'in1k', 240, 0.882, 200, '[rwightman/pyt...models]*'),  # noqa
    ('efficientnet_edge_large_b', '0491', 'd502326f9568f096491354a117f12562cf47e038', 'v0.0.434', 'EfficientNet-Edge-Large-b', '1905.11946', 'in1k', 300, 0.904, 90, '[rwightman/pyt...models]*'),  # noqa
]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError("Pretrained model for {name} is not available.".format(name=model_name))
    error, sha1_hash, repo_release_tag, _, _, _, _, _, _, _ = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join("~", ".torch", "models")):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.
    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = "{name}-{error}-{short_sha1}.pth".format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning("Mismatch in the content of model file detected. Downloading again.")
    else:
        logging.info("Model file not found. Downloading to {}.".format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + ".zip"
    _download(
        url="{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip".format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError("Downloaded file has different hash. Please try again.")


def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """
    Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. " \
            "Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning("File {} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the `repo_url` is overridden, consider switching to "
                                      "the default repo.".format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, "s" if retries > 1 else ""))

    return fname


def _check_sha1(file_name, sha1_hash):
    """
    Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    file_name : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(file_name, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def load_model(net,
               file_path):
    """
    Load model state dictionary from a file.
    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import torch

    import warnings
    from collections import OrderedDict

    pretrained_dict = torch.load(file_path)
    model_dict = net.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    net.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'
        )
    else:
        print('Successfully loaded pretrained weights')
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )
    return net

def download_model(net,
                   model_name,
                   local_model_store_dir_path=os.path.join("~", ".torch", "models")):
    """
    Load model state dictionary from a file with downloading it if necessary.
    Parameters
    ----------
    net : Module
        Network in which weights are loaded.
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    net = load_model(
        net=net,
        file_path=get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path))

    return net

def calc_num_params(net):
    """
    Calculate the count of trainable parameters for a model.
    Parameters
    ----------
    net : Module
        Analyzed model.
    """
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count
