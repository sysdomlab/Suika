# This file is not included in the original Metis

# replace args with fixed hyperparameters

min_group_scale_variance = 0.5
max_permute_len = 4
# max_permute_len = 10

## deprecated, we must serve different tasks and models, so these two values are deduced at read_profile time
# max_profiled_tp_degree = 4
# max_profiled_batch_size = 4

from Suika.external.Metis.utils import metis_logger
metis_logger.info(f"hyperparameters: min_group_scale_variance {min_group_scale_variance}, max_permute_len {max_permute_len}")