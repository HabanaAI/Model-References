import argparse
import os
import sys
from menta.src import backend as K
import menta.dlo as menta_dlo
from stereo.models.menta.menta_utils import generate_menta_checkpoint


def run_dlo_main():
    try:
        menta_dlo.main()
    except Exception as e:
        K.clear_session()
        raise e
    K.clear_session()


def main():
    parser = argparse.ArgumentParser(description="Start stereo DNN training job on SageMaker")
    parser.add_argument('-o', '--output_path', type=str, help="quantization path")
    parser.add_argument('-m', '--model', type=str,
                        help="model name (mutually exclusive with menta_model argument)")
    parser.add_argument('-mm', '--menta_model', type=str,
                        help="directory of the menta model (mutually exclusive with model argument)")
    parser.add_argument('-r', '--rerun_menta', action="store_true",
                        help="regenerate MENTA checkpoint even if already exists")
    parser.add_argument('--aws', action="store_true")
    parser.add_argument('--instance_type', choices=["ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p3.16xlarge"],
                        default="ml.p3.2xlarge")

    args, unknown_args = parser.parse_known_args()

    if args.model and args.menta_model:
        raise ValueError("Use either model or menta_model")
    if args.model:
        if args.aws:
            s3_menta_model = os.path.join("s3://mobileye-habana/mobileye-team-stereo", "menta_models", args.model)
        else:
            s3_menta_model = None
        menta_model = generate_menta_checkpoint(args.model, s3_dst=s3_menta_model,
                                                use_existing_model=not args.rerun_menta)
    elif args.menta_model:
        menta_model = args.menta_model
    else:
        raise ValueError("Specify model or menta_model")

    sys.argv = [menta_dlo.__file__,
                "quantize",
                "-mc",
                menta_model,
                "-o",
                args.output_path,
                "-j",
                os.path.join(os.path.dirname(__file__), "quantization_template.json")]
    sys.argv.extend(unknown_args)

    if args.aws:
        os.environ['AWS_ROLE'] = 'sagemaker-stereo'
        sys.argv.extend(["--aws", "--instance_type", args.instance_type])

    run_dlo_main()


if __name__ == '__main__':
    main()
