from lans import *
from firedrake import *

nx = 100
ny = 30
Lx = nx/ny
Ly = 1.0

mesh = RectangleMesh(nx, ny, Lx, Ly)

uv_normal_dirichlet_bdys = {1:as_vector([1,0]),
                            3:as_vector([0,0]),
                            4:as_vector([0,0])}

u_tangential_dirichlet_bdys = {3:as_vector([0,0]),
                               4:as_vector([0,0])}
v_tangential_dirichlet_bdys = {3:as_vector([0,0]),
                               4:as_vector([0,0])}

v_inflow_bdys = {1:as_vector([1,0])}
v_not_inflow_bdys = [2,3,4]

stepper = \
    LANS_timestepper(mesh=mesh, degree=1, gamma=1.0e5, alpha=0.1, nu=0.1,
                     dt=0.01,
                     uv_normal_dirichlet_bdys=uv_normal_dirichlet_bdys,
                     u_tangential_dirichlet_bdys=u_tangential_dirichlet_bdys,
                     v_tangential_dirichlet_bdys=v_tangential_dirichlet_bdys,
                     v_inflow_bdys=v_inflow_bdys,
                     v_not_inflow_bdys=v_not_inflow_bdys,
                     advection=True)

stepper.run(tmax=10., dumpt=0.01,
            filename='channel', verbose=True)
