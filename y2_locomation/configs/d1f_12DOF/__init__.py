from utils.task_registry import task_registry

from .d1f_12DOF_flat_config import *
from .d1f_12DOF_climb_config import *
task_registry.register("d1f_12DOF_flat",D1F12DOFFlat,D1F12DOFFlatCfg(),D1F12DOFFlatCfgPPO())
task_registry.register("d1f_12DOF_climb",D1F12DOFClimb,D1F12DOFClimbCfg(),D1F12DOFClimbCfgPPO())
