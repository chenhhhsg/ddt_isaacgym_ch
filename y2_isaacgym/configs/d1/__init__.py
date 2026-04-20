from utils.task_registry import task_registry
from .d1_rough_config import *
task_registry.register("d1_rough",D1Rough,D1RoughCfg(),D1RoughCfgPPO())  
task_registry.register("d1_rough_play",D1Rough,D1RoughCfg_Play(),D1RoughCfgPPO())  

from .d1_flat_config import *
task_registry.register("d1_flat",D1Flat,D1FlatCfg(),D1FlatCfgPPO())
task_registry.register("d1_flat_play",D1Flat,D1FlatCfg_Play(),D1FlatCfgPPO())

from .d1_flat_height_config import *
task_registry.register("d1_flat_height",D1FlatHeight,D1FlatHeightCfg(),D1FlatHeightCfgPPO())