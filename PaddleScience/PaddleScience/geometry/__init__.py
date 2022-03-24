# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from .rectangular import Rectangular   #当执行包含中间圆柱时用该模块,对应的模型是rec_with_cyl_jk

from .cylinder_in_cube import CylinderInRectangular
# from .rectangular_backup import Rectangular  # 默认的LDC几何外形，对应的模型是rec_only

# 修改几何尺寸1，使用rectangular_use_cir ,RE_4 
# from .rectangular_use_cir import Rectangular



# 修改几何尺寸2，使用rectangular_use_cirRE_100,使用的是uniform离散节点
# from .rectangular_use_cirRE_100 import Rectangular


# 使用sampling随机生成节点，且圆周与四周节点固定，使用
from .rectangular_use_cirRE_100_new import Rectangular


