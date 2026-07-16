from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .l0.l0_config import L0Cfg, L0CfgPPO
from .l0.l0_env import L0Env
from .l1.l1_config import L1Cfg, L1CfgPPO
from .l1.l1_env import L1Env
from .CAS02.CAS02_LB_config import CAS02_LBCfg, CAS02_LBCfgPPO
from .CAS02.CAS02_LB_env import CAS02_LBEnv
from .l13dof.l13dof_config import L13dofCfg, L13dofCfgPPO
from .l13dof.l13dof_env import L13dofEnv

from humanoid.utils.task_registry import task_registry

task_registry.register( "l0", L0Env, L0Cfg(), L0CfgPPO() )
task_registry.register( "l1", L1Env, L1Cfg(), L1CfgPPO() )
task_registry.register( "CAS02_LB", CAS02_LBEnv, CAS02_LBCfg(), CAS02_LBCfgPPO() )
task_registry.register( "l13dof", L13dofEnv, L13dofCfg(), L13dofCfgPPO() )




