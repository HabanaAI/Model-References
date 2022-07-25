import os


def set_env_params():
    # Temporary workaround for HPU device in order to support bicubic interpolate mode.
    # It will be removed in a subsequent release.
    os.environ['ENABLE_EXPERIMENTAL_FLAGS'] = "1"
    os.environ['MAX_RMW_TENSOR_BYTES'] = "524288"


def setup_hpu(args):
    if args.device == 'hpu':
        if not args.lazy_mode:
            os.environ["PT_HPU_LAZY_MODE"] = "2"
        set_env_params()
        from habana_frameworks.torch.utils.library_loader import load_habana_module
        load_habana_module()
        if hasattr(args, "use_hmp") and args.use_hmp:
            from habana_frameworks.torch.hpex import hmp
            hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                        fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)


def mark_step(args):
    if args.device == 'hpu' and args.lazy_mode:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
