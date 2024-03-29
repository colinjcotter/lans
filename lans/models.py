from lans import forms
import firedrake as fd

class StokesAlpha(object):

    def __init__(self, mesh, degree, gamma, alpha, nu, F,
                 u_dirichlet_bcs, w_dirichlet_bcs):
        """:arg mesh: the mesh to solve on
        :arg degree: integer, the degree of the BDM space.
        :arg gamma: float, the interior penalty stabilisation parameter
        :arg alpha: float, the LANS lengthscale
        :arg nu: float, the kinematic viscosity
        :arg F: vector-valued expression giving the source term

        :arg u_dirichlet_bcs: dictionary with keys being tags
        of boundaries where Dirichlet conditions are enforced
        for u and values being expressions for the boundary values

        :arg w_dirichlet_bcs: dictionary with keys being tags
        of boundaries where Dirichlet conditions are enforced
        for w and values being expressions for the boundary values
        """

        gamma = fd.Constant(gamma)
        alpha = fd.Constant(alpha)
        nu = fd.Constant(nu)
        
        V = fd.FunctionSpace(mesh, "BDM", degree+1)
        VV = fd.VectorFunctionSpace(mesh, "BDM", degree+1, dim=2)
        #VV contains u-w
        QQ = fd.VectorFunctionSpace(mesh, "DG", degree, dim=2)
        #QQ contains p-q
        W = VV * QQ

        U = fd.Function(W) #the solution
        
        self.U = U

        uu, pp = fd.split(U)
        u = uu[0, :]
        w = uu[1, :]
        p = pp[0]
        q = pp[1]

        duu, dpp = fd.TestFunctions(W)
        du = duu[0, :]
        dw = duu[1, :]
        dp = dpp[0]
        dq = dpp[1]
        
        #continuity equation
        eqn = dp*fd.div(u)*fd.dx
        
        #circulation equation
        #pressure stabilisation
        eqn += gamma*fd.div(du)*fd.div(u)*fd.dx
        #pressure gradient term
        eqn -= fd.div(du)*p*fd.dx
        #viscosity
        eqn += nu*fd.inner(du, w)*fd.dx
        eqn += nu*forms.get_laplace_form(mesh, w, du, alpha,
                                         dirichlet_bdys=
                                         w_dirichlet_bcs)
        #source term
        eqn -= fd.inner(F, du)*fd.dx

        #vector equation in Stokes operator
        #viscous operator
        eqn += forms.get_laplace_form(mesh, u, dw, nu,
                                      dirichlet_bdys=
                                      u_dirichlet_bcs)
        #pressure gradient term
        eqn -= fd.div(dw)*q*fd.dx
        #the w bit
        eqn -= fd.inner(w, dw)*fd.dx
        #pressure stabilisation
        eqn += gamma*fd.div(dw)*fd.div(w)*fd.dx
        #divergence constraint for viscosity
        eqn += fd.div(w)*dq*fd.dx

        bcs = []
        for bdy, bvalue in u_dirichlet_bcs.items():
            #u bc conditions
            bcs.append(fd.DirichletBC(W.sub(0).sub(0), bvalue, bdy))

        for bdy, bvalue in w_dirichlet_bcs.items():
            #w bc conditions
            bcs.append(fd.DirichletBC(W.sub(0).sub(1), bvalue, bdy))

        lans_prob = fd.NonlinearVariationalProblem(eqn, U, bcs=bcs)

        sparameters = {
            "mat_type":"matfree",
            'snes_monitor': None,
            "ksp_type": "fgmres",
            "ksp_gmres_modifiedgramschmidt": None,
            'ksp_monitor': None,
            "ksp_rtol": 1e-8,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_off_diag_use_amat": True,
        }

        bottomright = {
            "ksp_type": "gmres",
            "ksp_max_it": 3,
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
            "Mp_pc_type": "bjacobi",
            "Mp_sub_pc_type": "ilu"
        }

        sparameters["fieldsplit_1"] = bottomright

        topleft_LU = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "lu",
            "assembled_pc_factor_mat_solver_type": "umfpack"
        }
        sparameters["fieldsplit_0"] = topleft_LU
        ctx = {"mu": nu/gamma/2}

        luparameters={
            'snes_type':'ksponly',
            'ksp_type':'gmres',
            'ksp_monitor':None,
            'ksp_atol':1.0e-50,
            'ksp_rtol':1.0e-10,
            'mat_type':'aij',
            'pc_type':'lu',
            "pc_factor_mat_solver_type": "mumps"}

        vbasis = fd.VectorSpaceBasis(constant=True)
        nullspace = fd.MixedVectorSpaceBasis(W, [W.sub(0), vbasis])
        self.solver = fd.NonlinearVariationalSolver(lans_prob,
                                                    solver_parameters=sparameters,
                                                    nullspace=nullspace,
                                                    appctx=ctx)

    def run(self, filename):
        f = fd.File(filename+'.pvd')
        uu, pp = self.U.split()
        u = uu.sub(0)
        w = uu.sub(1)
        p = pp.sub(0)
        q = pp.sub(1)
        self.solver.solve()
        f.write(u, w, p, q)

class LANS_timestepper(object):

    def __init__(self, mesh, degree, gamma, alpha, nu, dt,
                 uv_normal_dirichlet_bdys, u_tangential_dirichlet_bdys,
                 v_tangential_dirichlet_bdys,
                 v_inflow_bdys, v_not_inflow_bdys):
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

        gamma = fd.Constant(gamma)
        alpha = fd.Constant(alpha)
        nu = fd.Constant(nu)
        self.dt = dt
        dt = fd.Constant(dt)
        beta = fd.Constant(1.0) #advection scaling parameter
        
        V = fd.FunctionSpace(mesh, "BDM", degree)
        VV = fd.VectorFunctionSpace(mesh, "BDM", degree, dim=3)
        #VV contains u-v-w
        QQ = fd.VectorFunctionSpace(mesh, "DG", degree-1, dim=2)
        #QQ contains p-q
        W = VV * QQ

        wn = fd.Function(W) #state at timestep n
        wnp = fd.Function(W) #state at timestep n+1

        self.wn = wn
        self.wnp = wnp

        uun, pn = fd.split(wn)
        un = uun[0, :]
        uunp, pnp = fd.split(wnp)
        unp = uunp[0, :]
        vnh = uunp[1, :]
        wnh = uunp[2, :]
        pnh = pnp[0]
        qnh = pnp[1]

        unh = (un + unp)/2
        
        duu, dpp = fd.TestFunctions(W)
        du = duu[0, :]
        dv = duu[1, :]
        dw = duu[2, :]
        dp = dpp[0]
        dq = dpp[1]
        
        #u-v relation
        eqn = fd.inner(du, unp - vnh)*fd.dx
        eqn += forms.get_laplace_form(mesh, unp, du, alpha**2,
                                      dirichlet_bdys=
                                      u_tangential_dirichlet_bdys,
                                      surface=True)

        #continuity equation
        eqn += dp*fd.div(unp)*fd.dx
        
        #circulation equation
        #pressure stabilisation
        eqn += gamma*fd.div(dv)*fd.div(unp)*fd.dx
        #pressure gradient term (we scale pressure by dt)
        eqn -= fd.div(dv)*pnh*fd.dx
        #time-derivative
        eqn += fd.inner(dv, unp - un)*fd.dx
        eqn += forms.get_laplace_form(mesh, unp - un,
                                      du, alpha**2,
                                      dirichlet_bdys=
                                      u_tangential_dirichlet_bdys)
        #advection
        eqn += dt*forms.get_advective_form(mesh, unp, vnh, dv,
                                           v_inflow_bdys,
                                           v_not_inflow_bdys)
        #viscosity
        eqn += dt*nu*fd.inner(dv, wnh)*fd.dx
        eqn += dt*forms.get_laplace_form(mesh, wnh, dv, nu,
                                         dirichlet_bdys=
                                         v_tangential_dirichlet_bdys)

        #vector equation in Stokes operator
        eqn += forms.get_laplace_form(mesh, unh, dw, nu,
                                         dirichlet_bdys=
                                         v_tangential_dirichlet_bdys)
        eqn -= fd.div(dw)*qnh*fd.dx
        #pressure stabilisation
        eqn += gamma*fd.div(dw)*fd.div(wnh)*fd.dx
        #divergence constraint for viscosity
        eqn += fd.div(wnh)*dq*fd.dx

        bcs = []
        d = wn.geometric_dimension()
        for bdy, bvalue in uv_normal_dirichlet_bdys.items():
            #u bc conditions
            bcs.append(fd.DirichletBC(W.sub(0).sub(0), bvalue, bdy))
            #w bc conditions
            bcs.append(fd.DirichletBC(W.sub(0).sub(2), bvalue, bdy))

        lans_prob = fd.NonlinearVariationalProblem(eqn, wnp, bcs=bcs)

        sparameters = {
            "mat_type":"matfree",
            'snes_monitor': None,
            "ksp_type": "fgmres",
            "ksp_gmres_modifiedgramschmidt": None,
            'ksp_monitor': None,
            "ksp_rtol": 1e-8,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_off_diag_use_amat": True,
        }

        bottomright = {
            "ksp_type": "gmres",
            "ksp_max_it": 3,
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
            "Mp_pc_type": "bjacobi",
            "Mp_sub_pc_type": "ilu"
        }

        sparameters["fieldsplit_1"] = bottomright

        topleft_LU = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "lu",
            "assembled_pc_factor_mat_solver_type": "mumps"
        }
        sparameters["fieldsplit_0"] = topleft_LU
        ctx = {"mu": nu/gamma/2}

        sparameters={'ksp_type':'preonly',
                     'mat_type':'aij',
                     'pc_type':'lu',
                     "pc_factor_mat_solver_type": "mumps"}

        self.solver = fd.NonlinearVariationalSolver(lans_prob,
                                                    solver_parameters=sparameters,
                                                    appctx=ctx)

    def run(self, tmax, dumpt, filename, verbose=False):
        t = 0.
        tdump = 0.
        f = fd.File(filename+'.pvd')

        uun, pn = self.wn.split()
        un = uun.sub(0)
        vn = uun.sub(1)
        self.wnp.assign(self.wnp)
        
        while t < tmax - self.dt/2:
            t += self.dt
            if verbose:
                print(t, self.dt, tmax)
                print(fd.norm(fd.div(un)), "div")
            self.solver.solve()
            self.wn.assign(self.wnp)

            tdump += self.dt
            if tdump > dumpt - self.dt/2:
                tdump -= dumpt
                f.write(un, vn, pn)
