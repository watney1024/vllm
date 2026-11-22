"""pytest 公共配置：把 ``lesson/routeep`` 加入 sys.path，使 ``import routeep`` 可用。"""

import pathlib
import sys

_ROUTEPP_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_ROUTEPP_DIR) not in sys.path:
    sys.path.insert(0, str(_ROUTEPP_DIR))
