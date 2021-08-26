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

L = 1; H = 1;
#Gmsh mesh.
mesh = Mesh()
with XDMFFile("mesh/test_3.xdmf") as infile: #very_fine
    infile.read(mesh)
cell_size = mesh.hmax()

# elastic parameters
mu = 80.77 #Ambati et al
lambda_ = 121.15 #Ambati et al
Gc = Constant(2.7e-3)
ell = Constant(5*cell_size)

bnd_facets = MeshFunction("size_t", mesh,1)
bnd_facets.set_all(0)

# Sub domain for BC
def right(x, on_boundary):
    return near(x[0], L) and on_boundary

def top(x, on_boundary):
    return near(x[1], H/2) and on_boundary

def down(x, on_boundary):
    return near(x[1], -H/2) and on_boundary

traction_boundary_1 = AutoSubDomain(top)
traction_boundary_1.mark(bnd_facets, 41)
traction_boundary_2 = AutoSubDomain(down)
traction_boundary_2.mark(bnd_facets, 42)
right = AutoSubDomain(right)
right.mark(bnd_facets, 43)
ds = Measure('ds')(subdomain_data=bnd_facets)

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
    return 2.0*mu*(eps(u)) + lambda_*tr(eps(u))*Identity(2)

def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return (a(alpha))*sigma_0(u)

z = sympy.Symbol("z")
c_w = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
Gc_eff = Gc * (1 + cell_size/(ell*4*float(c_w)))

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
bc1 = DirichletBC(V_u.sub(0), u_D, bnd_facets, 41)
bc2 = DirichletBC(V_u, Constant((0,0)), bnd_facets, 42)
bc_u = [bc1, bc2]

# Damage
bc_alpha1 = DirichletBC(V_alpha, Constant(0), bnd_facets, 41)
bc_alpha2 = DirichletBC(V_alpha, Constant(0), bnd_facets, 42)
bc_alpha3 = DirichletBC(V_alpha, Constant(0), bnd_facets, 43)
bc_alpha = [bc_alpha1, bc_alpha2, bc_alpha3]

import ufl
E_du = ufl.replace(E_u,{u:du})
problem_u = LinearVariationalProblem(lhs(E_du), rhs(E_du), u, bc_u)
solver_u = LinearVariationalSolver(problem_u)
solver_u.parameters.update({"linear_solver" : "mumps"})

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
solver_alpha.getKSP().setType('cg')
solver_alpha.getKSP().setTolerances(rtol=1e-8)
solver_alpha.getKSP().getPC().setType('lu')
solver_alpha.getKSP().setFromOptions()
solver_alpha.setTolerances(rtol=1e-5,atol=1e-8,max_it=2000)
solver_alpha.setFromOptions()
b = Function(V_alpha).vector()
solver_alpha.setFunction(pb_alpha.F, b.vec())
A = PETScMatrix()
solver_alpha.setJacobian(pb_alpha.J, A.mat())

#Putting crack in
c = 1.7e-2
B = 1/c - 1
d = Expression('x[0] < 0.5*L ? fabs(x[1]) : sqrt(x[1]*x[1] + (x[0]-0.5*L)*(x[0]-0.5*L))', L=L, degree=1)
#crack = Expression('d < ell ? 1 : 0', d=d, ell=ell, degree=1)
crack = Expression('d < ell ? B*0.25*Gc/ell*(1-d/ell) : 0', d=d, ell=ell, B=B, Gc=Gc, degree = 1) #see in Hughes
lb = interpolate(crack, V_alpha)
#lb = interpolate(Constant(0), V_alpha)
alpha.vector()[:] = lb.vector()
alpha.vector().apply('insert')
ub = interpolate(Constant(1), V_alpha) # upper bound, set to 1
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())
  
file_alpha = File("alpha.pvd") 
file_u = File("u.pvd")

#do a solve in alpha with u=0 before starting?
solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
xx = alpha.copy(deepcopy=True)
xv = as_backend_type(xx.vector()).vec()
#solver_alpha.solve(None, xv)
#alpha.vector()[:] = xv
#alpha.vector().apply('insert')

#Then solve in u and compute the force

solver_u.solve()
file_u << u
file_alpha << alpha

#Force imposée
v = Expression(('(x[1]+0.5*L)/L', '0'), L=L, degree=1)
v = interpolate(v, V_u)
load = inner(dot(sigma_0(u), n), as_vector((1,0))) * ds(41)
print(assemble(load))
