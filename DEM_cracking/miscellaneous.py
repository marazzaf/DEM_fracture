# coding: utf-8
from dolfin import *
import numpy as np
from numpy.linalg import norm
from scipy.sparse import dok_matrix

def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver"""
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

def mass_matrix(mesh_, d_, dim_, rho_, nb_ddl_ccG_):
    if d_ == dim_: #vectorial problem
        U_DG = VectorFunctionSpace(mesh_, "DG", 0)
        one = Constant(np.ones(dim_)) #to get the diagonal
    elif d_ == 1: #scalar problem
        U_DG = FunctionSpace(mesh_, "DG", 0)
        one = Constant(1.) #to get the diagonal

    #original mass matrix
    u_DG = TrialFunction(U_DG)
    v_DG = TestFunction(U_DG)
    M = rho_ * inner(u_DG,v_DG) * dx
    res = assemble(action(M, one)).get_local()
    res.resize(nb_ddl_ccG_)

    return res

def DEM_interpolation(func, problem):
    """Interpolates a function or expression to return a DEM vector containg the interpolation."""

    return problem.DEM_to_DG.T * local_project(func, problem.DG_0).vector().get_local() + problem.DEM_to_CG.T * local_project(func, problem.CG).vector().get_local()

def Dirichlet_BC(form, DEM_to_CG):
    L = assemble(form)
    return DEM_to_CG.T * L.get_local()

def assemble_volume_load(load, problem):
    v = TestFunction(problem.DG_0)
    form = inner(load, v) * dx
    L = assemble(form)
    return problem.DEM_to_DG.T * L


def schur_matrices(nb_ddl_cells_, nb_ddl_ccG_):
    aux = list(np.arange(nb_ddl_cells_))
    aux_bis = list(np.arange(nb_ddl_cells_, nb_ddl_ccG_))

    #Get non Dirichlet values
    mat_not_D = sp.dok_matrix((nb_ddl_cells_, nb_ddl_ccG_))
    for (i,j) in zip(range(mat_not_D.shape[0]),aux):
        mat_not_D[i,j] = 1.

    #Get Dirichlet boundary conditions
    mat_D = sp.dok_matrix((nb_ddl_ccG_ - nb_ddl_cells_, nb_ddl_ccG_))
    for (i,j) in zip(range(mat_D.shape[0]),aux_bis):
        mat_D[i,j] = 1.
    return mat_not_D.tocsr(), mat_D.tocsr()
