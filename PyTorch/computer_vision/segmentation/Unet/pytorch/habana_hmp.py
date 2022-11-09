from typing import Optional, Union

from .misc import _HPU_AVAILABLE
if _HPU_AVAILABLE:
    from habana_frameworks.torch.hpex import hmp

class HPUPrecision:
    """Plugin that enables bfloat/half support on HPUs.

    Args:
        precision: The precision to use.
        opt_level: Choose optimization level for hmp.
        bf16_file_path: Path to bf16 ops list in hmp O1 mode.
        fp32_file_path: Path to fp32 ops list in hmp O1 mode.
        verbose: Enable verbose mode for hmp.
    """

    def __init__(
        self,
        precision: Union[str, int],
        opt_level: str = "O2",
        bf16_file_path: Optional[str] = None,
        fp32_file_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if not _HPU_AVAILABLE:
            raise Exception("HPU precision plugin requires HPU devices.")
        if precision not in ('16','32', 'bf16'): #supported_precision_values
            raise ValueError(
                f"`Trainer(accelerator='hpu', precision={precision!r})` is not supported."
                f" `precision` must be one of: 16, 32, bf16."
            )
        self.precision = precision
        if precision in ("16", "bf16"):
            hmp.convert(
                opt_level=opt_level, bf16_file_path=bf16_file_path, fp32_file_path=fp32_file_path, isVerbose=verbose
            )
