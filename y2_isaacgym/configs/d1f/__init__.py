from utils.task_registry import task_registry

from .d1f_flat_config import *
task_registry.register("d1f_flat",D1FFlat,D1FFlatCfg(),D1FFlatCfgPPO())
task_registry.register("d1f_flat_play",D1FFlat,D1FFlatCfg_Play(),D1FFlatCfgPPO())

from .d1f_climb_config import *
task_registry.register("d1f_climb",D1FClimb,D1FClimbCfg(),D1FClimbCfgPPO())
