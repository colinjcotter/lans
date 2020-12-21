"""
Forms for Lagrangian-averaged Navier Stokes models.
"""

import firedrake as fd

def mycross(u, v):
    """
    Consistent cross product in 2D or 3D so we can write one equation
    set for both.

    :arg u: Expression
    :arg v: Expression

    :return w: Expression
    """

    d = u.geometric_dimension()

    #in the below "scalar" means scalar multiplied by unit vector
    #in the z direction
    if d == 2:
        if fd.as_ufl(u).ufl_shape == (2,):
            if fd.as_ufl(v).ufl_shape == (2,):
                #vector cross vector
                return u[0]*v[1] - u[1]*v[0]
            else:
                assert(fd.as_ufl(v).ufl_shape == ()):
                #vector cross scalar
                return fd.as_vector((u[1]*v, -u[0]*v))
        else:
            assert(fd.as_ufl(u).ufl_shape == ())
            assert(fd.as_ufl(v).ufl_shape == (2,))
            #scalar cross vector
            return fd.as_vector((u*v[1], u*v[0]))
    else:
        assert(d==3)
        return fd.cross(u, v)

def mycurl(u):
    """
    Consistent curl in 2D or 3D so we can write one equation
    set for both.

    :arg u: Expression
    :return w: Expression
    """

    d = u.geometric_dimension()

    #in the below "scalar" means scalar multiplied by unit vector
    #in the z direction
    if d == 2:
        if fd.as_ufl(u).ufl_shape == (2,):
            #vector curl
            return u[1].dx(0) - u[0].dx(1)
        else:
            assert(fd.as_ufl(v).ufl_shape == ()):
            #scalar curl
            return fd.as_vector((u.dx(1), -u.dx(0)))
    else:
        assert(d==3)
        return fd.curl(u)

def get_laplace_form(u, du, kappa, nu=10, dirichlet_bdys={},
                     use_bvalue=False):
    """Return a UFL form for the (minus, i.e. positive-definite) Laplacian
    plus boundary integrals

    :arg u: Trial function, or Coefficient
    :arg du: TestFunction
    :arg kappa: diffusion tensor
    :arg nu: Float, stabilisation constant
    :arg dirichlet_bdys: dictionary with keys being tags of boundaries
    where Dirichlet conditions are enforced (for tangential cpts of
    vectors since normal components are enforced strongly) and values
    being expressions for the boundary values
    :arg use_bvalue: logical, if true, use the given boundary value 
    in the boundary integral (non-symmetric) otherwise use the unknown.
    """

    if fd.as_ufl(kappa).ufl_shape == ():
        kappa = fd.Identity*kappa

    mesh = du.functionspace().mesh
    n = fd.FacetNormal(mesh)
    nu = fd.Constant(nu)
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)

    lf = fd.inner(fd.grad(du), fd.dot(fd.kappa, fd.grad(u)))*fd.dx
    #interior facet terms
    #consistent term
    lf -= fd.inner(2*fd.avg(fd.out(du, n)),
                   2*fd.avg(fd.dot(fd.kappa, fd.grad(u))))*fd.dS
    #symmetric term
    lf -= fd.inner(2*fd.avg(fd.out(u, n)),
                   2*fd.avg(fd.dot(fd.kappa, fd.grad(du))))*fd.dS
    #penalty term
    lf += nu*fd.inner(fd.jump(u)/h, fd.dot(fd.avg(kappa), fd.jump(du)))*fd.dS

    #exterior facet terms (for Dirichlet bcs for tangential components)
    for bdy, bvalue in bdys.items():
        #consistent term
        lf -= fd.inner(fd.out(du, n),
                       fd.dot(fd.kappa, fd.grad(u)))*fd.dS(bdy)
        #symmetric term
        if use_bvalue:
            lf -= fd.inner(fd.out(bvalue, n),
                           fd.dot(fd.kappa, fd.grad(du)))*fd.dS(bdy)
        if use_bvalue:
            lf -= fd.inner(fd.out(u, n),
                           fd.dot(fd.kappa, fd.grad(du)))*fd.dS(bdy)
        #penalty term
        lf += nu*fd.inner((u0-bvalue)/h,
                          fd.dot(kappa, du))*fd.ds(bdy)

    return lf
