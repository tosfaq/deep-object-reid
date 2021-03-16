import json
import numpy as np
import os
import sys
from PIL import Image

from torchreid.data.transforms import build_inference_transform

def wrap_nncf_model(model, cfg, datamanager_for_init, nncf_config_path):
    import nncf
    from nncf import (NNCFConfig, create_compressed_model,
                      register_default_init_args)
    from nncf.initialization import InitializingDataLoader
    from nncf.dynamic_graph.input_wrapping import nncf_model_input

    if nncf_config_path:
        with open(nncf_config_path) as f:
            nncf_config_data = json.load(f)
    else:
        # TODO(lbeynens): remove this, this is a DEBUG feature for compatibility with older
        #                 experiments
        nncf_config_data = {
#            "input_info": {
#                "sample_size": [1, 3, h, w]
#                },
            "compression": [
                {
                    "algorithm": "quantization",
                    "initializer": {
                        "range": {
                            "num_init_samples": 8192, # Number of samples from the training dataset
                                                      # to consume as sample model inputs for purposes of setting initial
                                                      # minimum and maximum quantization ranges
                            },
                        "batchnorm_adaptation": {
                            "num_bn_adaptation_samples": 8192, # Number of samples from the training
                                                               # dataset to pass through the model at initialization in order to update
                                                               # batchnorm statistics of the original model. The actual number of samples
                                                               # will be a closest multiple of the batch size.
                            #"num_bn_forget_samples": 1024, # Number of samples from the training
                                                            # dataset to pass through the model at initialization in order to erase
                                                            # batchnorm statistics of the original model (using large momentum value
                                                            # for rolling mean updates). The actual number of samples will be a
                                                            # closest multiple of the batch size.
                            }
                        }
                    }
                ],
            "log_dir": "."
        }

    h, w = cfg.data.height, cfg.data.width
    nncf_config_data.setdefault('input_info', {})
    nncf_config_data['input_info']['sample_size'] = [1, 3, h, w]

    nncf_config = NNCFConfig(nncf_config_data)
    print(f'nncf_config =\n{nncf_config}')

    train_loader = datamanager_for_init.train_loader
    class ReidInitializeDataLoader(InitializingDataLoader): #TODO: check is it correct
        def get_inputs(self, dataloader_output):
            # define own InitializingDataLoader class using approach like
            # parse_data_for_train and parse_data_for_eval in the class Engine
            # dataloader_output[0] should be image here
            args = (dataloader_output[0], )
            return args, {}

    cur_device = next(model.parameters()).device
    print(f'cur_device = {cur_device}')

    # TODO: add `if not loading pretrained model` here
    wrapped_loader = ReidInitializeDataLoader(train_loader)
    nncf_config = register_default_init_args(nncf_config, wrapped_loader, device=cur_device)

    transform = build_inference_transform(
        cfg.data.height,
        cfg.data.width,
        norm_mean=cfg.data.norm_mean,
        norm_std=cfg.data.norm_std,
    )
    def random_image(height, width):
        if True: ### DEBUG #######################################################
            print(':::DEBUG: random_image call')
            import traceback
            traceback.print_stack(file=sys.stdout)
        input_size = (height, width, 3)
        img = np.random.rand(*input_size).astype(np.float32)
        img = np.uint8(img * 255)

        out_img = Image.fromarray(img)

        return out_img

    def dummy_forward(model):
        prev_training_state = model.training
        model.eval()
        input_img = random_image(cfg.data.height, cfg.data.width)
        input_blob = transform(input_img).unsqueeze(0)
        assert len(input_blob.size()) == 4
        input_blob = input_blob.to(device=cur_device)
        input_blob = nncf_model_input(input_blob)
        model(input_blob)
        model.train(prev_training_state)

    # TODO: think if this is required
    #       (NNCF has the default wrap_inputs builder)
    def wrap_inputs(args, kwargs):
        assert not kwargs
        assert len(args) == 1
        return (nncf_model_input(args[0]), ), {}

    model.dummy_forward_fn = dummy_forward
    if 'log_dir' in nncf_config:
        os.makedirs(nncf_config['log_dir'], exist_ok=True)
    print(f'nncf_config["log_dir"] = {nncf_config["log_dir"]}')

    resuming_state_dict = None #TODO
    compression_ctrl, model = create_compressed_model(model,
                                                      nncf_config,
                                                      dummy_forward_fn=dummy_forward,
                                                      wrap_inputs_fn=wrap_inputs,
                                                      resuming_state_dict=resuming_state_dict)
    return compression_ctrl, model


