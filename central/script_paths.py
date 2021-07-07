###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

"""Contains the framework-specific look-up tables for model-specific main training script"""
from pathlib import Path

def get_tensorflow_script_path(model: str) -> Path:
    """Return path to script for available TensorFlow models"""
    main_fw_dir = Path(__file__).parent.parent.joinpath('TensorFlow')
    model_to_path = {
        "resnet_estimator": main_fw_dir / "computer_vision" / "Resnets" / "imagenet_main.py",
        "resnet_keras": main_fw_dir / "computer_vision" / "Resnets" / "resnet_keras" / "resnet_ctl_imagenet_main.py",
        "densenet_keras": main_fw_dir / "computer_vision" / "densenet_keras" / "train.py",
        "albert": main_fw_dir / "nlp" / "albert" / "demo_albert.py",
        "bert": main_fw_dir / "nlp" / "bert" / "demo_bert.py",
        "efficientdet": main_fw_dir / ".." / "staging" / "TensorFlow" / "computer_vision" / "efficientdet" / "demo_efficientdet.py",
        "unet2d": main_fw_dir / "computer_vision" / "Unet2D" / "unet2d_demo.py",
        "maskrcnn": main_fw_dir / "computer_vision" / "maskrcnn" / "demo_mask_rcnn.py",
        "ssd_resnet34": main_fw_dir / "computer_vision" / "SSD_ResNet34" / "demo_ssd.py",
        "transformer": main_fw_dir / "nlp" / "transformer" / "demo_transformer.py",
        "mobilenet_v2": main_fw_dir / ".." / "staging" / "TensorFlow" / "computer_vision" / "mobilenetv2" / "research" / "slim" / "train_image_classifier.py",
        "pacman_segnet": main_fw_dir / ".." / "internal" / "TensorFlow" / "computer_vision" / "pacman_segnet" / "train.py",
        "vgg_segnet": main_fw_dir / ".." / "internal" / "TensorFlow" / "computer_vision" / "Segnet" / "keras_segmentation" / "__main__.py"
    }
    assert model in model_to_path, f"TensorFlow model {model} not available, please provide one of {model_to_path.keys()}"
    return model_to_path[model]

def get_pytorch_script_path(model: str) -> Path:
    """Return path to script for available PyTorch models"""
    main_fw_dir = Path(__file__).parent.parent.joinpath('PyTorch')
    model_to_path = {
        "dlrm": main_fw_dir / "recommendation" / "dlrm" / "demo_dlrm.py",
        "bert": main_fw_dir / "nlp" / "bert" / "demo_bert.py",
        "resnet50": main_fw_dir / "computer_vision" / "ImageClassification" / "ResNet" / "demo_resnet.py",
    }
    assert model in model_to_path, f"PyTorch model {model} not available, please provide one of {model_to_path.keys()}"
    return model_to_path[model]

def get_script_path(framework: str, model: str) -> Path:
    """Return path to script for available models for the specified framework"""
    framework_specific_paths = {
        "tensorflow": get_tensorflow_script_path,
        "pytorch": get_pytorch_script_path,
    }
    return framework_specific_paths[framework](model)
