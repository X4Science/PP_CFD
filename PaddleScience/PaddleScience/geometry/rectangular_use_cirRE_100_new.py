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

from .geometry_discrete import GeometryDiscrete
from .geometry import Geometry
import numpy as np
import vtk
import matplotlib.pyplot as plt
import math
from .new_geo_add import NGD
import pdb



# pdb.set_trace()
# Rectangular
class Rectangular(Geometry):
    """
    Two dimentional rectangular

    Parameters:
        space_origin: cordinate of left-bottom point of rectangular
        space_extent: extent of rectangular

    Example:
        >>> import paddlescience as psci
        >>> geo = psci.geometry.Rectangular(space_origin=(0.0,0.0), space_extent=(1.0,1.0))

    """

    # init function
    def __init__(self,
                 time_dependent=False,
                 time_origin=None,
                 time_extent=None,
                 space_origin=None,
                 space_extent=None):
        super(Rectangular, self).__init__(time_dependent, time_origin,
                                          time_extent, space_origin,
                                          space_extent)

        # check time inputs 
        if (time_dependent == True):
            if (time_origin == None or not np.isscalar(time_origin)):
                print("ERROR: Please check the time_origin")
                exit()
            if (time_extent == None or not np.isscalar(time_extent)):
                print("ERROR: Please check the time_extent")
                exit()
        else:
            if (time_origin != None):
                print(
                    "Errror: The time_origin need to be None when time_dependent is false"
                )
                exit()
            if (time_extent != None):
                print(
                    "Errror: The time_extent need to be None when time_dependent is false"
                )
                exit()

        # check space inputs and set dimension
        self.space_origin = (space_origin, ) if (
            np.isscalar(space_origin)) else space_origin
        self.space_extent = (space_extent, ) if (
            np.isscalar(space_extent)) else space_extent

        lso = len(self.space_origin)
        lse = len(self.space_extent)
        self.space_ndims = lso
        if (lso != lse):
            print(
                "ERROR: Please check dimention of space_origin and space_extent."
            )
            exit()
        elif lso == 1:
            self.space_shape = "rectangular_1d"
        elif lso == 2:
            self.space_shape = "rectangular_2d"
        elif lso == 3:
            self.space_shape = "rectangular_3d"
        else:
            print("ERROR: Rectangular supported is should be 1d/2d/3d.")

    # domain sampling discretize
    def sampling_discretize(self, time_nsteps=None, space_point_size=None, space_nsteps=None):
        # TODO
        # check input
        self.space_point_size = (space_point_size, ) if (
            np.isscalar(space_point_size)) else space_point_size

        self.space_nsteps = (space_nsteps, ) if (
            np.isscalar(space_nsteps)) else space_nsteps

        # discretization time space with linspace
        # steps = []
        if self.time_dependent == True:
            time_steps = np.linspace(
                self.time_origin, self.time_extent, time_nsteps, endpoint=True)

        # sampling in space discretization
        space_points = []
        # pdb.set_trace()
        for i in range(space_point_size):
            current_point = []
            for j in range(2):
                # get a random value in [space_origin[j], space_extent[j]]
                random_value = self.space_origin[j] + (
                        self.space_extent[j] - self.space_origin[j]
                ) * np.random.random_sample()
                current_point.append(random_value)
            space_points.append(current_point)
        space_points = np.array(space_points)       
        # pdb.set_trace()

        # 定义圆
        theta = 0
        cir_center = (0, 0)
        x_circle = []
        y_circle = []
        theta = 0
        r = 0.02
        while theta <= 2 * math.pi:
            x_circle = np.append(x_circle, r * math.cos(theta) + cir_center[0])
            y_circle = np.append(y_circle, r * math.sin(theta) + cir_center[1])
            theta += 0.01
        x_c1 = x_circle.reshape(len(x_circle), 1)
        y_c1 = y_circle.reshape(len(y_circle), 1)
        xy_circle = np.hstack([x_c1, y_c1])
        # pdb.set_trace()
        # 删除圆内的节点
        index_o = 0
        index_i = 0
        bc_inner = []
        for point in space_points:
            if math.sqrt((point[0] - cir_center[0]) * (point[0] - cir_center[0]) + (point[1] - cir_center[1]) * (
                    point[1] - cir_center[1])) <= r:
                index_i = index_o
                bc_inner.append(index_i)
            index_o += 1
        bc_inner_list = np.unique(bc_inner)
        # print(bc_inner_list.shape)
        space_points_del = np.delete(space_points, bc_inner_list, axis=0)
        # print(space_points_del.shape[0])
        space_points_new = np.vstack((space_points_del, xy_circle))
        space_size = space_points_new.shape[0]
        # pdb.set_trace()
        # 1) 定义圆周边界节点编号
        nbc_cir = xy_circle.shape[0]
        # bc_index_cir = np.ndarray(nbc_cir, dtype=int)
        bc_index_cir = []
        i = 0
        while i < nbc_cir:
            bc_index_cir.append(space_points_del.shape[0] + i)
            i += 1

        # pdb.set_trace()
        # add boundry value
        if (self.space_ndims == 1):
            nbc = 2
            space_points.append(self.space_origin[-1])
            space_points.append(self.space_extent[-1])
            bc_index = np.ndarray(2, dtype=int)
            bc_index[0] = space_point_size
            bc_index[1] = space_point_size + 1
        elif (self.space_ndims == 2):
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nbc = nx * ny - (nx - 2) * (ny - 2)
            # bc_index = np.ndarray(nbc, dtype=int)
            bc_index_side = []
            nbc = 0
            x_start = self.space_origin[0]
            delta_x = (self.space_extent[0] - self.space_origin[0]) / (nx - 1)
            y_start = self.space_origin[1]
            delta_y = (self.space_extent[1] - self.space_origin[1]) / (ny - 1)
            for j in range(ny):
                for i in range(nx):
                    if (j == 0 or j == ny - 1 or i == 0 or i == nx - 1):
                        x = x_start + i * delta_x
                        y = y_start + j * delta_y
                        space_points_new = np.append(space_points_new, [[x, y]], axis=0)
                        # bc_index_side[nbc] = space_points_new.shape[0] + nbc
                        bc_index_side.append(space_size + nbc)
                        nbc += 1
        elif (self.space_ndims == 3):
            #当前用不到，不用管
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            nbc = nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
            bc_index = np.ndarray(nbc, dtype=int)
            nbc = 0
            x_start = self.space_origin[0]
            delta_x = (self.space_extent[0] - self.space_origin[0]) / (nx - 1)
            y_start = self.space_origin[1]
            delta_y = (self.space_extent[1] - self.space_origin[1]) / (ny - 1)
            z_start = self.space_origin[2]
            delta_z = (self.space_extent[2] - self.space_origin[2]) / (nz - 1)
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        if (k == 0 or k == nz - 1 or j == 0 or j == ny - 1 or
                                i == 0 or i == nx - 1):
                            x = x_start + i * delta_x
                            y = y_start + j * delta_y
                            z = z_start + k * delta_z
                            space_points.append([x, y, z])
                            bc_index[nbc] = space_point_size + nbc
                            nbc += 1
        space_domain = space_points_new
        # pdb.set_trace()
        # 3)定义完整的bc_index,包含圆与四个边界
        nbc = len(bc_index_cir) + len(bc_index_side)
        bc_index_cir = np.array(bc_index_cir)
        bc_index_side = np.array(bc_index_side)
        base_bc_index = np.vstack((bc_index_cir.reshape(-1, 1), bc_index_side.reshape(-1, 1)))
        bc_index = np.ndarray(nbc, dtype=int)
        for i in range(len(base_bc_index)):
            bc_index[i] = base_bc_index[i][0]
        
        # pdb.set_trace()
      # bc_index with time-domain
        nbc = len(bc_index)
        if self.time_dependent == True:
            bc_offset = np.arange(time_nsteps).repeat(len(bc_index))
            bc_offset = bc_offset * len(space_domain)
            bc_index = np.tile(bc_index, time_nsteps)
            bc_index = bc_index + bc_offset

        # pdb.set_trace()
        # IC index
        if self.time_dependent == True:
            ic_index = np.arange(len(space_domain))

        # return discrete geometry
        geo_disc = GeometryDiscrete()
        domain = []
        if self.time_dependent == True:
            # Get the time-space domain which combine the time domain and space domain
            for time in time_steps:
                current_time = time * np.ones(
                    (len(space_domain), 1), dtype=np.float32)
                current_domain = np.concatenate(
                    (current_time, space_domain), axis=-1)
                domain.append(current_domain.tolist())
            time_size = len(time_steps)
            space_domain_size = space_domain.shape[0]
            domain_dim = len(space_domain[0]) + 1
            domain = np.array(domain).reshape(
                (time_size * space_domain_size, domain_dim))

        if self.time_dependent == True:
            geo_disc.set_domain(
                time_domain=time_steps,
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent,
                time_space_domain=domain)
            geo_disc.set_bc_index(bc_index)
            geo_disc.set_ic_index(ic_index)
        else:
            geo_disc.set_domain(
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent)
            geo_disc.set_bc_index(bc_index)

        vtk_obj_name, vtk_obj, vtk_data_size = self.obj_vtk()
        geo_disc.set_vtk_obj(vtk_obj_name, vtk_obj, vtk_data_size)

        return geo_disc


    # domain discretize
    def discretize(self, time_nsteps=None, space_nsteps=None):

        # check input
        self.space_nsteps = (space_nsteps, ) if (
            np.isscalar(space_nsteps)) else space_nsteps

        # steps1 = []
        if self.time_dependent == True:
            time_steps = np.linspace(
                self.time_origin, self.time_extent, time_nsteps, endpoint=True)
        
        #######################
        #计算bc_index与space_domain
        # bc_index, space_domain = NGD.bc_index_calc(self)


        # region1
        space_origin_1 = (-0.04, -0.04)
        space_extent_1 = (0.04, 0.04)
        space_nsteps1 = (81, 81)
        nx1 = space_nsteps1[0]  # x方向离散有301个点
        ny1 = space_nsteps1[1]  # y方向离散有201个点


        # region2：右侧区域网格，x方向长度为0.4，离散11个点，y方向宽度为0.4，离散11个点
        space_origin_2 = (0.04, self.space_origin[1])
        space_extent_2 = self.space_extent
        space_nsteps2 = (61, 61)
        nx2 = space_nsteps2[0]  # x方向离散有101个点
        ny2 = space_nsteps2[1]  # y方向离散有101个点

        # region3：上侧网格网格粗，x方向长度为0.3，离散11个点，y方向宽度为0.1，离散5个点
        space_origin_3 = (-0.04, 0.04)
        space_extent_3 = (0.05, self.space_extent[1])
        space_nsteps3 = (31, 21)
        nx3 = space_nsteps3[0]  # x方向离散有XX个点
        ny3 = space_nsteps3[1]  # y方向离散有XX个点

        # region4：下侧网格网格粗，x方向长度为0.3，离散11个点，y方向宽度为0.1，离散5个点
        space_origin_4 = (-0.04, self.space_origin[1])
        space_extent_4 = (0.04, -0.04)
        space_nsteps4 = (31, 21)
        nx4 = space_nsteps4[0]  # x方向离散有11个点
        ny4 = space_nsteps4[1]  # y方向离散有11个点

        # region5：左侧区域，网格粗，x方向长度为0.1，离散11个点，y方向宽度为0.4，离散11个点
        space_origin_5 = self.space_origin
        space_extent_5 = (-0.04, self.space_extent[1])
        space_nsteps5 = (31, 61)
        nx5 = space_nsteps5[0]  # x方向离散有11个点
        ny5 = space_nsteps5[1]  # y方向离散有11个点

        # 定义区域1的节点
        steps1 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps1.append(
                np.linspace(
                    space_origin_1[i],
                    space_extent_1[i],
                    space_nsteps1[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh1 = np.meshgrid(steps1[1], steps1[0], sparse=False, indexing='ij')
        # print(mesh1)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain1 = np.stack(
            (mesh1[1].reshape(-1), mesh1[0].reshape(-1)), axis=-1)
        space_domain1 = domain1
        # print(space_domain1.shape)

        theta = 0
        x_circle = []
        y_circle = []
        theta = 0
        r = 0.01
        while theta <= 2 * math.pi:
            x_circle = np.append(x_circle, r * math.cos(theta))
            y_circle = np.append(y_circle, r * math.sin(theta))
            theta += 0.0005
        x_c1 = x_circle.reshape(len(x_circle), 1)
        y_c1 = y_circle.reshape(len(y_circle), 1)
        xy_circle = np.hstack([x_c1, y_c1])

        # 删除domain中圆柱内部的坐标点
        index_o = 0
        index_i = 0
        bc_inner = []
        for point in space_domain1:
            for inner_p in xy_circle:
                if abs(inner_p[0]) - abs(point[0]) > 1e-5 and abs(inner_p[1]) - abs(point[1]) > 1e-5:
                    index_i = index_o
                    bc_inner.append(index_i)
            index_o += 1
        space_domain1 = np.delete(space_domain1, np.unique(bc_inner), axis=0)
        # print(space_domain1.shape)

        # 定义区域2的节点
        steps2 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps2.append(
                np.linspace(
                    space_origin_2[i],
                    space_extent_2[i],
                    space_nsteps2[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh2 = np.meshgrid(steps2[1], steps2[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain2 = np.stack(
            (mesh2[1].reshape(-1), mesh2[0].reshape(-1)), axis=-1)
        space_domain2 = domain2
        # print(space_domain2.shape)

        # 定义区域3的节点
        steps3 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps3.append(
                np.linspace(
                    space_origin_3[i],
                    space_extent_3[i],
                    space_nsteps3[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh3 = np.meshgrid(steps3[1], steps3[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain3 = np.stack(
            (mesh3[1].reshape(-1), mesh3[0].reshape(-1)), axis=-1)
        space_domain3 = domain3
        # print(space_domain3.shape)

        # 定义区域4的节点
        steps4 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps4.append(
                np.linspace(
                    space_origin_4[i],
                    space_extent_4[i],
                    space_nsteps4[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh4 = np.meshgrid(steps4[1], steps4[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain4 = np.stack(
            (mesh4[1].reshape(-1), mesh4[0].reshape(-1)), axis=-1)
        space_domain4 = domain4
        # print(space_domain4.shape)

        # 定义区域5的节点
        steps5 = []  # steps是在二维的平面内，将x方向从-0.05到0.05进行均匀离散，并记录所有的x方向的坐标，同时对于y方向也同样，最后得到包含两个数组的坐标集
        for i in range(self.space_ndims):
            steps5.append(
                np.linspace(
                    space_origin_5[i],
                    space_extent_5[i],
                    space_nsteps5[i],
                    endpoint=True))

        # 对x方向给定的向量与y方向给定的向量进行网格化，得到内部坐标（uniform形式）,mesh中两个矩阵，分别代表x，y
        mesh5 = np.meshgrid(steps5[1], steps5[0], sparse=False, indexing='ij')
        # print(mesh2)
        # 构建网格区域，将mesh中的两个矩阵展平，并整合成一组完整的坐标(x,y)，对应的是domain[index][0]表示网格点的x位置，domain[index][1]表示网格点的y位置。
        domain5 = np.stack(
            (mesh5[1].reshape(-1), mesh5[0].reshape(-1)), axis=-1)
        space_domain5 = domain5
        # print(space_domain5.shape)

        space_domain = np.vstack((space_domain1, space_domain2, space_domain3, space_domain4, space_domain5))
        # sys_domain = np.unique(sys_domain)
        # print(space_domain.shape)

        space_domain = np.array(list(set([tuple(t) for t in space_domain])))
        # print(f'space_domian.shape:{space_domain.shape}')

        # plt.scatter(space_domain[:, 0], space_domain[:, 1])
        # plt.show()

        # 定义完整bc_index:
        # 1) 定义圆柱周边边界
        nbc = 0
        id_o = 0
        index_c = 0
        bc_cir = []
        for pp in space_domain:
            for bp in xy_circle:
                if abs(bp[0] - pp[0]) <= 0.002 and abs(bp[1] - pp[1]) <= 0.002:
                    index_c = id_o
                    bc_cir.append(index_c)
            id_o += 1
        bc_cir = np.unique(bc_cir)
        nbc += len(bc_cir)  # 记录当前bc_index中包含的元素数量,代表圆柱边界的点数量
        # pdb.set_trace()

        # print("test"*3)

        # 2) 定义外围边界
        id_r = 0
        index_b = 0
        bc_side = []
        for pp in space_domain:
            if abs(space_origin_5[0] - pp[0]) < 0.0001 or abs(space_extent_2[0] - pp[0]) < 0.0001 or abs(
                    space_origin_5[1] - pp[1]) < 0.0001 or abs(space_extent_2[1] - pp[1]) < 0.0001:
                index_b = id_r
                bc_side.append(index_b)
            id_r += 1
        bc_side = np.unique(bc_side)

        # print("tttt"*4)
        # 定义边界点总数
        nbc += len(bc_side)

        # 定义bc_index,指的是domain中点的id数组
        base_bc_index = np.vstack((bc_side.reshape(-1, 1), bc_cir.reshape(-1, 1)))
        # pdb.set_trace()
        
        bc_index = np.ndarray(nbc, dtype=int)
        for i in range(len(base_bc_index)):
            bc_index[i] = base_bc_index[i][0]
        # pdb.set_trace()


        # new_geo_add(self.space_origin, self.space_extent, self.space_ndims)

        # meshgrid and stack to cordinates
        if self.time_dependent == True:
            nsteps = time_nsteps
            ndims = self.space_ndims + 1
        else:
            nsteps = 1
            ndims = self.space_ndims

        # bc_index with time-domain
        nbc = len(bc_index)
        if self.time_dependent == True:
            bc_offset = np.arange(time_nsteps).repeat(len(bc_index))
            bc_offset = bc_offset * len(space_domain)
            # pdb.set_trace()
            
            bc_index = np.tile(bc_index, time_nsteps)
            # pdb.set_trace()

            bc_index = bc_index + bc_offset
        p1 = bc_offset
        p2 = bc_index
        # pdb.set_trace()

        # IC index
        if self.time_dependent == True:
            ic_index = np.arange(len(space_domain))
            # pdb.set_trace()
        # return discrete geometry
        geo_disc = GeometryDiscrete()
        domain = []
        if self.time_dependent == True:
            # Get the time-space domain which combine the time domain and space domain
            for time in time_steps:
                current_time = time * np.ones(
                    (len(space_domain), 1), dtype=np.float32)
                current_domain = np.concatenate(
                    (current_time, space_domain), axis=-1)
                domain.append(current_domain.tolist())
            time_size = len(time_steps)
            space_domain_size = space_domain.shape[0]
            domain_dim = len(space_domain[0]) + 1
            domain = np.array(domain).reshape(
                (time_size * space_domain_size, domain_dim))
        tick_domain = domain  #### print domain
        sp_domain = space_domain
        # pdb.set_trace()
        if self.time_dependent == True:
            geo_disc.set_domain(
                time_domain=time_steps,
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent,
                time_space_domain=domain)
            geo_disc.set_bc_index(bc_index)
            geo_disc.set_ic_index(ic_index)
        else:
            geo_disc.set_domain(
                space_domain=space_domain,
                space_origin=self.space_origin,
                space_extent=self.space_extent)
            geo_disc.set_bc_index(bc_index)

        vtk_obj_name, vtk_obj, vtk_data_size = self.obj_vtk()
        geo_disc.set_vtk_obj(vtk_obj_name, vtk_obj, vtk_data_size)

        # mpl_obj, mpl_data_shape = self.obj_mpl()
        # geo_disc.set_mpl_obj(mpl_obj, mpl_data_shape)

        return geo_disc

    # visu vtk
    def obj_vtk(self):
        # prepare plane obj 2d
        if self.space_ndims == 2:
            vtkobjname = "vtkPlanceSource"
            self.plane = vtk.vtkPlaneSource()
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            self.plane.SetResolution(nx - 1, ny - 1)
            self.plane.SetOrigin(
                [self.space_origin[0], self.space_origin[1], 0])
            self.plane.SetPoint1(
                [self.space_extent[0], self.space_origin[1], 0])
            self.plane.SetPoint2(
                [self.space_origin[0], self.space_extent[1], 0])
            self.plane.Update()
            vtk_data_size = self.plane.GetOutput().GetNumberOfPoints()
            return vtkobjname, self.plane, vtk_data_size
        elif self.space_ndims == 3:
            vtkobjname = "vtkImageData"
            self.img = vtk.vtkImageData()
            self.img.SetOrigin(self.space_origin[0], self.space_origin[1],
                               self.space_origin[2])
            nx = self.space_nsteps[0]
            ny = self.space_nsteps[1]
            nz = self.space_nsteps[2]
            self.img.SetDimensions(nx, ny, nz)
            vtk_data_size = self.img.GetNumberOfPoints()
            return vtkobjname, self.img, vtk_data_size


# # visu matplotlib
# def obj_mpl(self):
#     # prepare plan obj 2d
#     if self.space_ndims == 2:
#         fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"})
#     return self.ax, (self.space_nsteps[0], self.space_nsteps[1])
