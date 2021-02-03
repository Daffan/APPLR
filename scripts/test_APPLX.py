import os
import sys
from os.path import join, dirname, abspath, exists
sys.path.append(dirname(dirname(abspath(__file__))))

from jackal_navi_envs.APPLX import APPLD_policy, APPLE_policy, APPLI_policy
APPLD_policy = APPLD_policy()
APPLE_policy = APPLE_policy()
APPLI_policy = APPLI_policy()
