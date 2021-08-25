#coding: utf-8

# Commented out IPython magic to ensure Python compatibility.
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
from ufl import sign
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

E, nu = Constant(1.0), Constant(0.3)
kappa = (3-nu)/(1+nu)
mu = 0.5*E/(1+nu)
Gc = Constant(1.5)
K1 = Constant(1.)
ell = Constant(5*cell_size)
h = mesh.hmax()

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)
cells_meshfunction = MeshFunction("size_t", mesh, 2)
dxx = dx(subdomain_data=cells_meshfunction)

class NotCrack(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0) or near(x[0], L) or near(x[1], -H/2) or near(x[1], H/2))
not_crack = NotCrack()
not_crack.mark(boundaries, 1)

V = FunctionSpace(mesh, 'CG', 1)
v = TestFunction(V)
A = FacetArea(mesh)
vec = assemble(v / A * ds(0)).get_local()
nz = vec.nonzero()[0]

#print(V.dofmap().global_dimension())
#sys.exit()

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
    mu    = E/(2.0*(1.0 + nu))
    lmbda = E*nu/(1 - nu) / (1 - 2*nu)
    return 2.0*mu*(eps(u)) + lmbda*tr(eps(u))*Identity(ndim)

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

#Dirichlet BC on disp
vel = 4 #given velocity
x = SpatialCoordinate(mesh)
r = Expression('sqrt((x[0]-v*t) * (x[0]-v*t) + x[1] * x[1])', v=vel, t=0, degree = 2)
X = x[1] / (x[0]-vel*r.t) #-l0
c_theta = 1. / sqrt(1 + X**2.) * sign(x[0]-vel*r.t) #-l0
c_theta_2 = sqrt(0.5 * (1+c_theta))
s_theta_2 = sqrt(0.5 * (1-c_theta)) * sign(x[1])
u_D = K1/(2*mu) * sqrt(r/(2*np.pi)) * (kappa - c_theta) * as_vector((c_theta_2,s_theta_2)) #condition de bord de Dirichlet en disp

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
bcalpha_1 = DirichletBC(V_alpha, Constant(1.0), boundaries, 0) #crack lips
#bcalpha_2 = DirichletBC(V_alpha, Constant(0.0), boundaries, 0)
bc_alpha = [bcalpha_0, bcalpha_1]

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
        alpha_error.vector()[:] = alpha.vector() - alpha_0.vector()
        err_alpha = norm(alpha_error.vector(),"linf")
        # monitor the results
        if MPI.comm_world.rank == 0:
            print("Iteration:  %2d, Error: %2.8g, alpha_max: %.8g" %(iter, err_alpha, alpha.vector().max()))
        # update iteration
        alpha_0.assign(alpha)
        iter=iter+1
    return (err_alpha, iter)

V_theta = VectorFunctionSpace(mesh, "CG", 1) #change with discretization?
theta = Function(V_theta, name="Theta")
theta_trial = TrialFunction(V_theta)
theta_test = TestFunction(V_theta)

d2v = dof_to_vertex_map(V_alpha)
xcoor = mesh.coordinates()[d2v][:, 0]

def find_crack_tip():
    # Estimate the current crack tip
    ind = alpha.vector().get_local() > 0.5
    if ind.any():
        xmax = xcoor[ind].max()
    else:
        xmax = 0.0
    x0 = MPI.max(MPI.comm_world, xmax)

    return [x0, 0]

def calc_theta(pos_crack_tip=[0., 0.]):
    x0 = pos_crack_tip[0]  # x-coordinate of the crack tip
    y0 = pos_crack_tip[1]  # y-coordinate
    r = 2*float(ell)
    R = 5*float(ell)

    def neartip(x, on_boundary):
        dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        return dist < r

    def outside(x, on_boundary):
        dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        return dist > R

    class bigcircle(SubDomain):
        def inside(self, x, on_boundary):
            dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
            return dist < 1.1*R

    bigcircle().mark(cells_meshfunction, 1)

    bc1 = DirichletBC(V_theta, Constant([1.0, 0.0]), neartip)
    bc2 = DirichletBC(V_theta, Constant([0.0, 0.0]), outside)
    bc = [bc1, bc2]
    a = inner(grad(theta_trial), grad(theta_test))*dx
    L = inner(Constant([0.0, 0.0]), theta_test)*dx
    solve(a == L, theta, bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})

def calc_gtheta():
    sig = sigma(u,alpha)
    psi = 0.5 * inner(sig, grad(u))

    # Static and dynamic energy release rates
    Gstat = inner(sig, dot(grad(u), grad(theta))) - psi*div(theta)

    # Damage dissipation rate
    q = 6*Gc*ell/8*grad(alpha)  # only for AT1!
    Gamma = (Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha))))*div(theta) - inner(q, grad(theta)*grad(alpha))

    # Generalized J-integral
    # Y = (1-self.problem.alpha)*inner(self.problem.material.sigma0(self.problem.u), self.problem.material.eps(self.problem.u))-3*Gc/(8*ell)
    Y = (1-alpha)*(Dx(u, 1))**2-3*Gc/(8*ell)
    JmG = (Y+div(q))*inner(grad(alpha), theta)

    Gstat_value = assemble(Gstat*dxx(1))
    Gamma_value = assemble(Gamma*dxx(1))
    JmG_value = assemble(JmG*dxx(1))

    return Gstat_value,Gamma_value,JmG_value


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

    #Energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)
    energies = [elastic_energy_value,surface_energy_value,elastic_energy_value+surface_energy_value]
    pos = find_crack_tip()
    calc_theta(pos)
    res = calc_gtheta()
    
    save_energies.write('%.2e %.5e %.5e %.5e %.5e %.5e %.5e %.4e\n' % (r.t, energies[0], energies[1], energies[2], res[0]/Gc_eff, res[1]/Gc_eff, res[2]/Gc_eff, pos[0]))
    

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

    #Checking BC
    #BC = project(u_D, V_u)
    #file_BC << (BC, rr.t)
    #u_D.t = rr.t
    # solve alternate minimization
    alternate_minimization(u,alpha,maxiter=30) #,alpha_0=alpha.copy(deepcopy=True))
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    postprocessing()
    #sys.exit()

save_energies.close()
