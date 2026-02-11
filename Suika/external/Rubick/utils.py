import collections

EnvParams = collections.namedtuple(
    "EnvParams",
    [
        "B_pcie",
        "B_nvlink",
    ],
)

# Due to the unit difference (Suika: seconds, Rubick: milliseconds), we keep most PerfParams in Rubick unit system
# and only convert it to seconds inside ThroughputFunction
PerfParams = collections.namedtuple(
    "PerfParams",
    [
        "k_bwd",
        "k_sync",
        "k_opt",
        "k_os_offload", # not used
        "k_const",
        "k_off", # not used
        "k_swap", # not used
    ],
)

from Suika.utils import get_logger
import logging
rubick_logger = get_logger("rubick", console_level=logging.ERROR)