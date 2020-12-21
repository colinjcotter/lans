import pytest
import firedrake as fd
import lans

def test_cross_2d():
    mesh = fd.UnitSquareMesh(3,3)
    x, y = fd.SpatialCoordinate(mesh)
    V = fd.FunctionSpace(mesh, "CG", 1)
    VV = fd.VectorFunctionSpace(mesh, "CG", 1)
    a_expr = fd.as_vector((fd.exp(fd.cos(fd.pi*x)*fd.cos(fd.pi*y)),
                        1))
    b_expr = fd.as_vector((2, (fd.cos(fd.pi*x)*fd.cos(fd.pi*y))**2))
    c_expr = fd.Constant(1)
    a = fd.Function(VV).interpolate(a_expr)
    b = fd.Function(VV).interpolate(b_expr)
    c = fd.Function(V).interpolate(c_expr)

    #doing a vector triple product
    d1 = fd.Function(V).assign(lans.mycross(a, lans.mycross(b, c)))
    d2 = fd.Function(V).assign(-fd.dot(a, b)*c)

    assert(fd.norm(d1-d2) < 1.0e-6)

@pytest.mark.parametrize('n, etol', [(10, 0.07),
                                     (20, 0.018),
                                     (40, 0.0046),
                                     (80, 0.0012)])
def test_laplacian(n, etol):
    mesh = fd.UnitSquareMesh(n, n)
    x, y = fd.SpatialCoordinate(mesh)
    VV = fd.VectorFunctionSpace(mesh, "CG", 1)
    rhs_exp = fd.as_vector(((2*fd.pi)**2*fd.cos(2*fd.pi*y),
                           2*(2*fd.pi)**2*fd.cos(2*fd.pi*x)*
                           fd.sin(2*fd.pi*y)))
    v = fd.Function(VV).interpolate(rhs_exp)
    dv = fd.TestFunction(VV)
    u = fd.Function(VV, name='usol')
    bvalue = fd.as_vector((fd.Constant(1), fd.Constant(0)))
    eqn = lans.get_laplace_form(u, dv, fd.Constant(1),
                                dirichlet_bdys={3:bvalue, 4:bvalue})
    eqn -= fd.inner(v, dv)*fd.dx
    bcs = [fd.DirichletBC(VV, bvalue, 3),
           fd.DirichletBC(VV, bvalue, 4)]
    vprob = fd.NonlinearVariationalProblem(eqn, u, bcs=bcs)
    vsolver = fd.NonlinearVariationalSolver(vprob,
                                           solver_parameters={
                                               'ksp_type':'preonly',
                                               'pc_type':'lu'})
    vsolver.solve()
    sol_exp = fd.as_vector((fd.cos(2*fd.pi*y),
                            fd.cos(2*fd.pi*x)*fd.sin(2*fd.pi*y)))
    usolve = fd.Function(VV, name='uexact').interpolate(sol_exp)
    assert(fd.norm(u-usolve) < etol)
