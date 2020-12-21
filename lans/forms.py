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
            assert(fd.as_ufl(v).ufl_shape == ())
            #scalar curl
            return fd.as_vector((u.dx(1), -u.dx(0)))
    else:
        assert(d==3)
        return fd.curl(u)

def get_laplace_form(u, du, kappa, dirichlet_bdys,
                     use_bvalue=False, rho=10):
    """
    Return a UFL form for the (minus, i.e. positive-definite) Laplacian
    plus boundary integrals

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

    mesh = du.function_space().mesh()
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
    return 2*avg(u)

def get_advective_form(u, v, w, inflow_bdys, not_inflow_bdys):
    """
    Return the ufl form for the Lie derivative of v with u
    
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

    mesh = w.function_space().mesh()
    n = fd.FacetNormal(mesh)
    Upwind = 0.5*(sign(dot(u, n))+1)

    #curl part of Lie derivative
    af = fd.inner(v, mycurl(mycross(u, w)))*fd.dx
    af -= fd.inner(both(Upwind*v),
                   both(mycross(n, mycross(u, w))))*fd.dS
    #grad part of Lie derivative (no facet terms as Hdiv test fn)
    af -= fd.div(w)*inner(v, u)*fd.dx
    
    #exterior facet integrals
    for bdy, bvalue in inflow_bdys.items():
        af -= fd.inner(bvalue, mycross(n, mycross(u, w)))*fd.ds(bdy)
        af -= fd.inner(w, n)*inner(bvalue, u)*fd.ds(bdy)
    for bdy, bvalue in not_inflow_bdys:
        af -= fd.inner(u, mycross(n, mycross(u, w)))*fd.ds(bdy)
        af -= fd.inner(w, n)*inner(v, u)*fd.ds(bdy)

    return af

def get_diamond(u, w, alpha):
    """
    Return the ufl form for the diamond operator for LANS
    
    :arg u: Expression for the advecting velocity
    :arg w: TestFunction
    :arg alpha: Expression for the alpha parameter
    """

    mesh = w.function_space().mesh()
    n = fd.FacetNormal(mesh)
    
    df = fd.div(w)*(inner(u,u)/2 + alpha**2*inner(grad(u), grad(u))/2)*fd.dx
    df -= fd.inner(w,n)*(inner(u,u)/2
                         + alpha**2*inner(grad(u), grad(u))/2)*fd.ds
    return df

class LANS_timestepper(object):

    def __init__(mesh, degree, gamma, alpha, nu, dt,
                 uv_normal_dirichlet_bdys, u_tangential_dirichlet_bdys,
                 v_tangential_dirichlet_bdys,
                 v_inflow_bdys, v_not_inflow_bdys)
        """:arg mesh: the mesh to solve on
        :arg degree: integer, the degree of the BDM space.
        :arg gamma: float, the interior penalty stabilisation parameter
        :arg alpha: float, the LANS lengthscale
        :arg nu: float, the kinematic viscosity
        :arg dt: float, the time step size

        :arg uv_normal_dirichlet_bdys: dictionary with keys being tags
        of boundaries where normal Dirichlet conditions are enforced
        for u and v and values being expressions for the boundary values
        (vector valued and the normal component is used, same value for
        u and v)

        :arg u_tangential_dirichlet_bdys: dictionary with keys being
        tags of boundaries where tangential Dirichlet conditions are
        enforced for u and values being expressions for the boundary
        values (vector valued and the tangential component is
        used). Assumes that normal component is being set
        consistently.

        :arg v_tangential_dirichlet_bdys: dictionary with keys being
        tags of boundaries where tangential Dirichlet conditions are
        enforced for v and values being expressions for the boundary
        values (vector valued and the tangential component is
        used). Assumes that normal component is being set
        consistently.

        :arg v_inflow_bdys: dictionary with keys being tags of
        boundaries where inflow Dirichlet conditions are enforced for
        v (for tangential cpts of vectors since normal components are
        enforced strongly) and values being expressions for the
        boundary values

        :arg v_not_inflow_bdys: list of tag of boundaries where inflow
        Dirichlet conditions are not enforced for v (for tangential
        cpts of vectors since normal components are enforced strongly)
        and values being expressions for the boundary values
        """

        gamma = Constant(gamma)
        alpha = Constant(alpha)
        nu = Constant(nu)
        self.dt = dt
        dt = Constant(dt)
        
        V = fd.FunctionSpace(mesh, "BDM", degree)
        VV = fd.VectorFunctionSpace(mesh, "BDM", degree, dim=2)
        Q = fd.FunctionSpace(mesh, "DG", degree-1)
        W = VV * Q

        wn = fd.Function(W) #state at timestep n
        wnp = fd.Function(W) #state at timestep n+1

        uun, pn = fd.split(wn)
        un = uun[0, :]
        vn = uun[1, :]
        uunp, pnp = fd.split(wnp)
        unp = uunp[0, :]
        vnp = uunp[1, :]

        uh = (un + unp)/2
        vh = (vn + vnp)/2
        
        duu, dp = fd.TestFunctions(W)
        du = duu[0, :]
        dv = duu[1, :]

        #u-v relation 
        eqn = fd.inner(du, unp - vnp)*dx
        eqn += get_laplace_form(u, du, alpha**2,
                                dirichlet_bdys=u_tangential_dirichlet_bdys)
        #pressure stabilisation
        eqn += gamma*fd.div(du)*fd.div(unp)*fd.dx

        #continuity equation
        eqn += dp*fd.div(unp)*fd.dx

        #time-derivative
        eqn += fd.inner(dv, vnp - vn)*dx

        #advection 
        eqn += dt*get_advective_form(unh, vnh, dv,
                                     v_inflow_bdys,
                                     v_not_inflow_bdys)

        #diamond
        eqn += dt*get_diamond(unh, dv, alpha)

        #viscosity (applied to v)
        eqn += dt*get_laplace_form(vnh, du, nu, v_tangential_dirichlet_bdys)

        bcs = []
        d = wn.geometric_dimension()
        for bdy, bvalue in dirichlet_bdys.items():
            if d == 2:
                bvalues = as_tensor([[bvalue[0], bvalue[1]],
                                     [bvalue[0], bvalue[1]]])
            else:
                bvalues = as_tensor([[bvalue[0], bvalue[1], bvalue[2]]
                                     [bvalue[0], bvalue[1], bvalue[2]]])
                bcs.append(DirichletBC(W.sub(0), bvalue, bdy))

        lans_prob = NonlinearVariationalProblem(eqn, wnp, bcs=bcs)
        lans_solver = NonlinearVariationalSolver(lans_prob,
                            NEED THE AUG LAG SOLVER PARAMETERS
