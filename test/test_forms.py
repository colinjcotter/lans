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
