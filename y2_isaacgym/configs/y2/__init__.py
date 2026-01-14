from utils.task_registry import task_registry

from .y2_flat_config import *
from .y2_climb_config import *
task_registry.register("y2_flat",Y2Flat,Y2FlatCfg(),Y2FlatCfgPPO())
task_registry.register("y2_climb",Y2Climb,Y2ClimbCfg(),Y2ClimbCfgPPO())
