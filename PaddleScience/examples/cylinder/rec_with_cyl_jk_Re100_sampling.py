# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddlescience as psci
import numpy as np
import pdb



def GenBC(xy, bc_index):
    tick_cir = []
    tick_side = []
    tick_inlet = []
    tick_outlet = []
    bc_value = np.zeros((len(bc_index) , 3)).astype(np.float32)   
    for i in range(len(bc_index)):
        id = bc_index[i] 
        if abs(xy[id][1] - (-0.2)) < 1e-4:
            bc_value[i][0] = 2.5   #Re=100
            bc_value[i][1] = 0.0
            tick_inlet.append('inelt')
        elif abs(xy[id][1] - 0.6) < 1e-4:
            bc_value[i][2] = 0
            tick_outlet.append('outlet')
        elif abs(xy[id][2] - 0.2) < 1e-4 or abs(xy[id][2] - (-0.2)) < 1e-4:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
            tick_side.append('side')
        else:
            bc_value[i][0] = 0.0
            bc_value[i][1] = 0.0
            tick_cir.append('cir')
    # pdb.set_trace()
    t_in = tick_inlet
    t_out = tick_outlet
    t_cir = tick_cir
    t_side = tick_side
    # pdb.set_trace()
    return bc_value

# Generate BC weight
def GenBCWeight(xy, bc_index):
    bc_weight = np.zeros((len(bc_index), 3)).astype(np.float32)
    for i in range(len(bc_index)):
        id = bc_index[i]
        if abs(xy[id][1] - (-0.2)) < 1e-4:
            bc_weight[i][0] = 1 - 5 * abs(xy[id][2])
            bc_weight[i][1] = 0.8
            bc_weight[i][2] = 0.8
            # print('find inlet')
        elif abs(xy[id][1] - 0.6) < 1e-4:
            bc_weight[i][0] = 0.8
            bc_weight[i][1] = 0.8
            bc_weight[i][2] = 0.8
            # print('find outlet ')
        elif abs(xy[id][2] - 0.2) < 1e-4 or abs(xy[id][2] - (-0.2)) < 1e-4:
            bc_weight[i][0] = 0.8
            bc_weight[i][1] = 0.8
            bc_weight[i][2] = 0.8
        else:
            bc_weight[i][0] = 0.8
            bc_weight[i][1] = 0.8
            bc_weight[i][2] = 0.8

    return bc_weight

# Generate IC value
def GenIC(txy, ic_index):
    ic_value = np.zeros((len(ic_index), 3)).astype(np.float32)
    for i in range(len(ic_index)):
        id = ic_index[i] 
        if abs(txy[id][0] - (-0.2)) < 1e-4:
            ic_value[i][0] = 2.5
            ic_value[i][1] = 0.0
        elif abs(txy[id][0] - 0.6) < 1e-4:
            ic_value[i][2] = 0
        else:
            ic_value[i][0] = 0.0
            ic_value[i][1] = 0.0
    return ic_value

################
# Generate IC weight
# def GenICWeight(txy, ic_index):
#     ic_weight = np.zeros((len(ic_index), 2)).astype(np.float32)
#     for i in range(len(ic_index)):
#         id = ic_index[i]
#         if abs(txy[id][2] - 0.05) < 1e-4:
#             ic_weight[i][0] = 1.0 - 20 * abs(txy[id][1])
#             ic_weight[i][1] = 1.0
#         else:
#             ic_weight[i][0] = 1.0
#             ic_weight[i][1] = 1.0
#     return ic_weight
################

if __name__ == "__main__":
    # Geometry
    geo = psci.geometry.Rectangular(
        time_dependent=True,
        time_origin=0,
        time_extent=1,
        space_origin=(-0.1, -0.1), 
        space_extent=(0.4, 0.1))

    # PDE Laplace
    pdes = psci.pde.NavierStokes(nu=1e-3, rho=1, dim=2, time_dependent=True)
    #使用标准单位m,s Re=(rho*D*V)/nu，D为圆柱直径0.04，V为来流速度，实际上nu=1e-3,rho=1000，当前为了验证模型收敛情况。

    # Discretization
    # pdes, geo = psci.discretize(
    #     pdes, geo, time_nsteps=5, space_nsteps=(101, 101))

    # Sampling discretization
    pdes, geo = psci.sampling_discretize(pdes, geo, time_nsteps= 50, space_point_size=5000, space_nsteps=(101, 101))

    # bc value
    bc_value = GenBC(geo.get_domain(), geo.get_bc_index())
    pdes.set_bc_value(bc_value=bc_value, bc_check_dim=[0, 1, 2])

    # ic value
    ic_value = GenIC(geo.get_domain(), geo.get_ic_index())
    pdes.set_ic_value(ic_value=ic_value, ic_check_dim=[0, 1, 2])

    # Network
    net = psci.network.FCNet(
        num_ins=3,
        num_outs=3,
        num_layers=10,
        hidden_size=50,
        dtype="float32",
        activation='tanh')

    # Loss, TO rename
    bc_weight = GenBCWeight(geo.domain, geo.bc_index)
    # ic_weight = GenBCWeight(geo.domain, geo.ic_index)
    loss = psci.loss.L2(pdes=pdes,
                        geo=geo,
                        eq_weight=0.01,
                        bc_weight=bc_weight,
                        synthesis_method='norm')

    # Algorithm
    algo = psci.algorithm.PINNs(net=net, loss=loss)

    # Optimizer
    opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

    # Solver
    solver = psci.solver.Solver(algo=algo, opt=opt)
    solution = solver.solve(num_epoch=100)

    # Use solution
    rslt = solution(geo)
    # Get the result of last moment
    rslt = rslt[(-geo.space_domain_size):, :]
    # u = rslt[:, 0]
    # v = rslt[:, 1]
    # u_and_v = np.sqrt(u * u + v * v)
    # psci.visu.save_vtk(geo, u, filename="rslt_u")
    # psci.visu.save_vtk(geo, v, filename="rslt_v")
    # psci.visu.save_vtk(geo, u_and_v, filename="u_and_v")
    psci.visu.save_vtk_points(filename="result", geo=geo, data=rslt)


'''
    openfoam_u = np.load("../ldc2d/openfoam/openfoam_u_100.npy")
    diff_u = u - openfoam_u
    RSE_u = np.linalg.norm(diff_u, ord=2)
    MSE_u = RSE_u * RSE_u / geo.get_domain_size()
    print("MSE_u: ", MSE_u)
    openfoam_v = np.load("../ldc2d/openfoam/openfoam_v_100.npy")
    diff_v = v - openfoam_v
    RSE_v = np.linalg.norm(diff_v, ord=2)
    MSE_v = RSE_v * RSE_v / geo.get_domain_size()
    print("MSE_v: ", MSE_v)
'''