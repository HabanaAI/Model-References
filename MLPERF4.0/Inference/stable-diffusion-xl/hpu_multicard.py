###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Added to support for multi-card inference
# - Added support for higher batch size inference
# - Added quatinzation support
# - Updated warmup implementation

import argparse
from flask import Flask, request, jsonify, make_response

import dataset
import coco
import torch
import os
import numpy as np

import habana_frameworks.torch.hpu as torch_hpu
import habana_frameworks.torch.core as htcore

import flask.cli
flask.cli.show_server_banner = lambda *args: None

import logging
logging.getLogger("werkzeug").disabled = True

import tools.generate_fp32_weights as gw

app = Flask(__name__)

# Initialize global vars
node = ""
model = None
ds = None
rank = 0
port = 0
warmup_samples = None

SUPPORTED_DATASETS = {
    "coco-1024": (
        coco.Coco,
        dataset.preprocess,
        coco.PostProcessCoco(),
        {"image_size": [3, 1024, 1024]},
    )
}

PORT = [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007]

@app.route('/predict/', methods=['POST'])
def predict():
    """Receives a query (e.g., a text) runs inference, and returns a prediction."""
    query_id = request.get_json(force=True)['id']
    data, _ = ds.get_samples(query_id)
    result = model.predict(data)
    processed_results = post_proc(result, query_id)
    result = np.stack(processed_results)
    response = make_response(result.tobytes())
    response.headers['output_shape'] = list(result.shape)
    return response

@app.route('/warmup/', methods=['POST'])
def warmup():
    """Receives a query (e.g., a text) runs inference, used for warmup only."""
    _ = model.predict(warmup_samples)
    return jsonify(response="Success")

@app.route('/getname/', methods=['POST', 'GET'])
def getname():
    """Returns the name of the SUT."""
    return jsonify(name=f'Demo SUT (Network SUT) node' + (' ' + node) if node else '')

def get_backend(backend, **kwargs):
    if backend == "pytorch":
        from backend_pytorch import BackendPytorch

        backend = BackendPytorch(**kwargs)

    elif backend == "debug":
        from backend_debug import BackendDebug

        backend = BackendDebug()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--backend", help="Name of the backend")
    parser.add_argument("--model-path", default=None, help="Path to model weights")
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="dtype of the model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to run the benchmark",
    )
    parser.add_argument(
        "--latent-framework",
        default="torch",
        choices=["torch", "numpy"],
        help="framework to load the latents",
    )
    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=1,
        help="max batch size in a single inference",
    )
    parser.add_argument('--hpu-graph',
        const=True,
        default=True,
        type=str2bool,
        nargs="?"
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=False,
        help="enable quantization"
    )
    args = parser.parse_args()

    node = 1
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # find backend
    backend = get_backend(
        args.backend,
        precision=args.dtype,
        device=args.device,
        model_path=args.model_path,
        batch_size=args.max_batchsize
    )
    htcore.hpu_set_env()
    model = backend.load()
    setattr(model.pipe, 'quantized', args.quantize)
    if args.quantize:
        # import quantization package and load quantization configuration
        try:
            from neural_compressor.torch.quantization import FP8Config, convert, prepare
        except ImportError:
            raise ImportError(
                "Module neural_compressor is missing. Please use a newer Synapse version to use quantization")

        # additional unet for last 2 steps
        import copy
        unet_bf16 = copy.deepcopy(backend.pipe.unet)

        if args.hpu_graph and torch_hpu.is_available():
            unet_bf16 = torch_hpu.wrap_in_hpu_graph(unet_bf16)
        setattr(backend.pipe, 'unet_bf16', unet_bf16)

        # replace bf16 weights to be quantized with fp32 weights
        temp_dict = gw.get_unet_weights(args.model_path)
        for name, module in backend.pipe.unet.named_modules():
            if name in temp_dict.keys():
                del module.weight
                setattr(module,'weight',torch.nn.Parameter(temp_dict[name].clone().to(args.device)))
        temp_dict.clear()
        quant_config_full_fp8 = os.getenv('QUANT_CONFIG')
        config_fp8 = FP8Config.from_json_file(quant_config_full_fp8)
        quant_config_partial_fp8 = os.getenv('QUANT_CONFIG_2')
        config_fp8_2 = FP8Config.from_json_file(quant_config_partial_fp8)
        backend.pipe.unet = convert(backend.pipe.unet, config_fp8)

        htcore.hpu_initialize(backend.pipe.unet, mark_only_scales_as_const=True)
        backend.pipe.unet_bf16 = convert(backend.pipe.unet_bf16, config_fp8_2)
        htcore.hpu_initialize(backend.pipe.unet_bf16, mark_only_scales_as_const=True)

    if args.hpu_graph and torch_hpu.is_available():
        backend.pipe.unet = torch_hpu.wrap_in_hpu_graph(backend.pipe.unet)
    # dataset to use
    dataset_class, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = dataset_class(
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_proc,
        pipe_tokenizer=model.pipe.tokenizer,
        pipe_tokenizer_2=model.pipe.tokenizer_2,
        latent_dtype=dtype,
        latent_device=args.device,
        latent_framework=args.latent_framework,
        **kwargs,
    )
    # Create synthetic samples for warmup
    syntetic_str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
    latents_pt= torch.rand(ds.latents.shape, dtype=dtype).to(args.device)
    warmup_samples = [
        {
            "input_tokens": ds.preprocess(syntetic_str, model.pipe.tokenizer),
            "input_tokens_2": ds.preprocess(syntetic_str, model.pipe.tokenizer_2),
            "latents":latents_pt,
        }
        for _ in range(args.max_batchsize)
    ]

    ds.load_query_samples(list(range(0, ds.get_item_count())))
    world_size = 1
    # Check for CUDA availability
    if torch.cuda.is_available():
        # Check for distributed environment variables
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1:
            # Initialize distributed process group
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(local_rank)
            rank = torch.distributed.get_rank()
    port = PORT[rank]
    app.run(debug=False, port=port)
