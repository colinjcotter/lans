from lans import *
from firedrake import *

nx = 30
ny = 30
Lx = 1.0
Ly = 1.0

mesh = RectangleMesh(nx, ny, Lx, Ly)

u_dirichlet_bcs = {1:as_vector([0.,0.]),
                   2:as_vector([0.,0.]),
                   3:as_vector([0.,0.]),
                   4:as_vector([1.,0.])}

w_dirichlet_bcs = {1:as_vector([0.,0.]),
                   2:as_vector([0.,0.]),
                   3:as_vector([0.,0.]),
                   4:as_vector([0.,0.])}
x, y = SpatialCoordinate(mesh)
F = as_vector([y*(1-y), 0])

model = StokesAlpha(mesh, degree=1, gamma=1.0e4, alpha=0.1, nu=0.1,
                    F=F, u_dirichlet_bcs=u_dirichlet_bcs,
                    w_dirichlet_bcs=w_dirichlet_bcs)
model.run('stokestest')
