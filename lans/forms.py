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
                assert(fd.as_ufl(v).ufl_shape == ())
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
            assert(fd.as_ufl(u).ufl_shape == ())
            #scalar curl
            return fd.as_vector((u.dx(1), -u.dx(0)))
    else:
        assert(d==3)
        return fd.curl(u)

def get_laplace_form(mesh, u, du, kappa, dirichlet_bdys,
                     use_bvalue=False, rho=10):
    """
    Return a UFL form for the (minus, i.e. positive-definite) Laplacian
    plus boundary integrals

    :arg mesh: the Mesh
    :arg u: Trial function, or Coefficient
    :arg du: TestFunction
    :arg kappa: diffusion tensor
    :arg dirichlet_bdys: dictionary with keys being tags of boundaries
    where Dirichlet conditions are enforced (for tangential cpts of
    vectors since normal components are enforced strongly) and values
    being expressions for the boundary values
    :arg use_bvalue: logical, if true, use the given boundary value 
    in the boundary integral (non-symmetric) otherwise use the unknown.
    :arg rho: Float, stabilisation constant
    """

    d = du.geometric_dimension()
    
    if fd.as_ufl(kappa).ufl_shape == ():
        if d == 2:
            kappa = fd.as_tensor([[kappa, 0],
                                  [0, kappa]])
        else:
            kappa = fd.as_tensor([[kappa, 0, 0],
                                  [0, kappa, 0],
                                  [0, 0, kappa]])

    n = fd.FacetNormal(mesh)
    rho = fd.Constant(rho)
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)

    lf = fd.inner(fd.grad(du), fd.dot(kappa, fd.grad(u)))*fd.dx
    #interior facet terms
    #consistent term
    lf -= fd.inner(2*fd.avg(fd.outer(du, n)),
                   2*fd.avg(fd.dot(kappa, fd.grad(u))))*fd.dS
    #symmetric term
    lf -= fd.inner(2*fd.avg(fd.outer(u, n)),
                   2*fd.avg(fd.dot(kappa, fd.grad(du))))*fd.dS
    #penalty term
    lf += rho*fd.inner(fd.jump(u)/h, fd.dot(fd.avg(kappa), fd.jump(du)))*fd.dS

    #exterior facet terms (for Dirichlet bcs for tangential components)
    for bdy, bvalue in dirichlet_bdys.items():
        #consistent term
        lf -= fd.inner(fd.outer(du, n),
                       fd.dot(kappa, fd.grad(u)))*fd.ds(bdy)
        #symmetric term
        if use_bvalue:
            lf -= fd.inner(fd.outer(bvalue, n),
                           fd.dot(kappa, fd.grad(du)))*fd.ds(bdy)
        if use_bvalue:
            lf -= fd.inner(fd.outer(u, n),
                           fd.dot(kappa, fd.grad(du)))*fd.ds(bdy)
        #penalty term
        lf += rho*fd.inner((u-bvalue)/h,
                          fd.dot(kappa, du))*fd.ds(bdy)

    return lf

def both(u):
    return 2*fd.avg(u)

def get_advective_form(mesh, u, v, w, inflow_bdys, not_inflow_bdys):
    """
    Return the ufl form for the Lie derivative of v with u
    
    : arg mesh: the Mesh
    :arg u: Expression for the advecting velocity
    :arg v: Expression for the circulation velocity
    :arg w: TestFunction
    :arg inflow_bdys: dictionary with keys being tags of boundaries
    where inflow Dirichlet conditions are enforced (for tangential cpts of
    vectors since normal components are enforced strongly) and values
    being expressions for the boundary values
    :arg not_inflow_bdys: list of tag of boundaries where inflow
    Dirichlet conditions are not enforced (for tangential cpts of
    vectors since normal components are enforced strongly) and values
    being expressions for the boundary values
    """

    n = fd.FacetNormal(mesh)
    Upwind = 0.5*(fd.sign(fd.dot(u, n))+1)

    #curl part of Lie derivative
    af = fd.inner(v, mycurl(mycross(u, w)))*fd.dx
    af -= fd.inner(both(Upwind*v),
                   both(mycross(n, mycross(u, w))))*fd.dS
    #grad part of Lie derivative (no facet terms as Hdiv test fn)
    af -= fd.div(w)*fd.inner(v, u)*fd.dx
    
    #exterior facet integrals
    for bdy, bvalue in inflow_bdys.items():
        af -= fd.inner(bvalue, mycross(n, mycross(u, w)))*fd.ds(bdy)
        af -= fd.inner(w, n)*fd.inner(bvalue, u)*fd.ds(bdy)
    for bdy in not_inflow_bdys:
        af -= fd.inner(u, mycross(n, mycross(u, w)))*fd.ds(bdy)
        af -= fd.inner(w, n)*fd.inner(v, u)*fd.ds(bdy)

    return af

def get_diamond(mesh, u, w, alpha):
    """
    Return the ufl form for the diamond operator for LANS
    
    :arg mesh: the Mesh
    :arg u: Expression for the advecting velocity
    :arg w: TestFunction
    :arg alpha: Expression for the alpha parameter
    """

    n = fd.FacetNormal(mesh)
    
    df = fd.div(w)*(fd.inner(u,u)/2
                    + alpha**2*fd.inner(fd.grad(u), fd.grad(u))/2)*fd.dx
    df -= fd.inner(w,n)*(fd.inner(u,u)/2
                         + alpha**2*fd.inner(fd.grad(u), fd.grad(u))/2)*fd.ds
    return df
