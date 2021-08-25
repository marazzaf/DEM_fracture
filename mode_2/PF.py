#coding: utf-8

# Commented out IPython magic to ensure Python compatibility.
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'viridis'
import sys, os, sympy, shutil, math
parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 2})
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

L = 5; H = 1;
#Gmsh mesh.
mesh = Mesh()
with XDMFFile("mesh/mesh_surfing_coarse.xdmf") as infile: #very_fine
    infile.read(mesh)
num_computation = 1
cell_size = mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions

# elastic parameters
mu = 80.77 #Ambati et al
lambda_ = 121.15 #Ambati et al
Gc = Constant(1.5)
ell = Constant(5*cell_size)
h = mesh.hmax()

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)
cells_meshfunction = MeshFunction("size_t", mesh, 2)
dxx = dx(subdomain_data=cells_meshfunction)

#Putting crack in

def w(alpha):
    """Dissipated energy function as a function of the damage """
    return alpha

def a(alpha):
    """Stiffness modulation as a function of the damage """
    k_ell = Constant(1.e-6) # residual stiffness
    return (1-alpha)**2+k_ell

def eps(u):
    """Strain tensor as a function of the displacement"""
    return sym(grad(u))

def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    return 2.0*mu*(eps(u)) + lambda_*tr(eps(u))*Identity(ndim)

def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return (a(alpha))*sigma_0(u)

z = sympy.Symbol("z")
c_w = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
Gc_eff = Gc * (1 + h/(ell*4*float(c_w)))

# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1)
V_alpha = FunctionSpace(mesh, "CG", 1)

# Define the function, test and trial fields
u, du, v = Function(V_u, name='disp'), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name='damage'), TrialFunction(V_alpha), TestFunction(V_alpha)

n = FacetNormal(mesh)

#Dirichlet BC
t_init = 9e-3
u_D = Expression('t*(x[1]+0.5*L)/L', L=L, t=t_init, degree=1)

#energies
elastic_energy = 0.5*inner(sigma(u,alpha), eps(u))*dx
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
total_energy = elastic_energy + dissipated_energy
# First directional derivative wrt u
E_u = derivative(total_energy,u,v)
# First and second directional derivative wrt alpha
E_alpha = derivative(total_energy,alpha,beta)
E_alpha_alpha = derivative(E_alpha,alpha,dalpha)

# Displacement
bc_u = DirichletBC(V_u, u_D, boundaries, 0) 

# Damage
bcalpha_0 = DirichletBC(V_alpha, Constant(0.0), boundaries, 1)
bc_alpha = [bcalpha_0]

import ufl
E_du = ufl.replace(E_u,{u:du})
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)
solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters.update({"linear_solver" : "umfpack"})

class DamageProblem():

    def f(self, x):
        """Function to be minimised"""
        alpha.vector()[:] = x
        return assemble(total_energy)

    def F(self, snes, x, F):
        """Gradient (first derivative)"""
        alpha.vector()[:] = x
        F = PETScVector(F)
        return assemble(E_alpha, tensor = F)
    
    def J(self, snes, x, J, P):
        """Hessian (second derivative)"""
        alpha.vector()[:] = x
        J = PETScMatrix(J)
        return assemble(E_alpha_alpha, tensor=J)

#creating solver damage
pb_alpha = DamageProblem()
solver_alpha = PETSc.SNES().create(PETSc.COMM_WORLD)
PETScOptions.set("snes_monitor")
solver_alpha.setTolerances(rtol=1e-8,max_it=200)
solver_alpha.setFromOptions()
solver_alpha.setType(PETSc.SNES.Type.VINEWTONSSLS)
#Putting the gradient and Hessian in the solver
b = Function(V_alpha).vector()
solver_alpha.setFunction(pb_alpha.F, b.vec())
A = PETScMatrix()
solver_alpha.setJacobian(pb_alpha.J, A.mat())

#Putting crack in
crack = Expression('x[0] < && fbas(x[1]) < 1 ? exp()  : 0', degree = 1) #see in Hughes
lb = interpolate(crack, V_alpha)

#bounds for damage
lb = interpolate(Constant("0."), V_alpha) # lower bound, initialize to 0
ub = interpolate(Constant("1."), V_alpha) # upper bound, set to 1
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())

def alternate_minimization(u,alpha,tol=1.e-5,maxiter=100,alpha_0=interpolate(Constant("0.0"), V_alpha)):
    # initialization
    iter = 1; err_alpha = 1
    alpha_error = Function(V_alpha)
    # iteration loop
    while err_alpha>tol and iter<maxiter:     
        # solve elastic problem
        solver_u.solve()
        # solve damage problem
        solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
        xx = alpha.copy(deepcopy=True)
        xv = as_backend_type(xx.vector()).vec()
        solver_alpha.solve(None, xv)
        alpha.vector()[:] = xv
        alpha.vector().apply('insert')
        alpha_error.vector()[:] = alpha.vector() - alpha_0.vector()
        alpha_error.vector().apply('insert')
        err_alpha = norm(alpha_error.vector(),"linf")
        # monitor the results
        if MPI.comm_world.rank == 0:
            print("Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" %(iter, err_alpha, alpha.vector().max()))
        # update iteration
        alpha_0.assign(alpha)
        iter=iter+1
    return (err_alpha, iter)

savedir = "ref_surfing_%i" % num_computation
if os.path.isdir(savedir):
    shutil.rmtree(savedir)    
file_alpha = File(savedir+"/alpha.pvd") 
file_u = File(savedir+"/u.pvd")
file_sig = File(savedir+"/sigma.pvd")
energies = []
save_energies = open(savedir+'/energies.txt', 'w')
file_BC = File(savedir+"/BC.pvd")

W = TensorFunctionSpace(mesh, 'DG', 0)
stress = Function(W, name="stress")

def postprocessing():
    ## Dump solution to file
    #file_alpha << (alpha,r.t)
    #file_u << (u,r.t)
    #stress.vector()[:] = project(sigma(u,alpha), W).vector()
    #file_sig << (stress,r.t)
    

T = 1 #final simulation time
dt = cell_size / 10 # / 5 # / 10

#Starting with crack lips already broken
aux = np.zeros_like(alpha.vector().get_local())
aux[nz] = np.ones_like(nz)
alpha.vector().set_local(aux)
lb.vector()[:] = alpha.vector() #irreversibility
file_alpha << (alpha,0)

while r.t < T:
    r.t += dt
    X = x[1] / (x[0]-vel*r.t)
    c_theta = 1. / sqrt(1 + X**2.) * sign(x[0]-vel*r.t)
    c_theta_2 = sqrt(0.5 * (1+c_theta))
    s_theta_2 = sqrt(0.5 * (1-c_theta)) * sign(x[1])
    u_D = K1/(2*mu) * sqrt(r/(2*np.pi)) * (kappa - c_theta) * as_vector((c_theta_2,s_theta_2))
    #u_D *= 1e-8
    bc_u = DirichletBC(V_u, u_D, boundaries, 0)
    problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)
    solver_u = LinearVariationalSolver(problem_u)
    solver_u.parameters.update({"linear_solver": "cg", "preconditioner": "hypre_amg"}) #{"linear_solver" : "umfpack"})

    # solve alternate minimization
    alternate_minimization(u,alpha,maxiter=30) #,alpha_0=alpha.copy(deepcopy=True))
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    lb.vector().apply('insert')
    postprocessing()
    #sys.exit()

save_energies.close()
