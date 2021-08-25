#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt
import sys

## Form compiler options
#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["optimize"] = True

#geomtry
L, H = 1, 1
mesh = Mesh()
with XDMFFile("test.xdmf") as infile:
    infile.read(mesh)

bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# elastic parameters
mu = 80.77 #Ambati et al
lambda_ = 121.15 #Ambati et al

#Dirichlet BC
t_init = 9e-3
u_D = Expression('t*(x[1]+0.5*L)/L', L=L, t=t_init, degree=1)

# Sub domain for BC
def top(x, on_boundary):
    return near(x[1], H/2) and on_boundary

def down(x, on_boundary):
    return near(x[1], -H/2) and on_boundary

bnd_facets.set_all(0)
traction_boundary_1 = AutoSubDomain(top)
traction_boundary_1.mark(bnd_facets, 41)
traction_boundary_2 = AutoSubDomain(down)
traction_boundary_2.mark(bnd_facets, 42)
ds = Measure('ds')(subdomain_data=bnd_facets)

#Elasticity
def sigma(v):
    return lambda_*div(v)*Identity(2) + 2*mu*sym(grad(v))

U = VectorFunctionSpace(mesh, 'CG', 1)
u = Function(U, name='disp')
u_ = TrialFunction(U)
v = TestFunction(U)

a = inner(sigma(u_),grad(v)) * dx
l = Constant(0) * v[0] * dx

#Dirichlet BC
bc_1 = DirichletBC(U.sub(0), u_D, bnd_facets, 41)
bc_2 = DirichletBC(U, Constant((0,0)), bnd_facets, 42)
bc = [bc_1, bc_2]


#solve
solve(a == l, u, bcs=bc)

img = plot(u[0])
plt.colorbar(img)
plt.savefig('disp_x.pdf')
#plt.show()
#sys.exit()

#Force imposée
v = Expression(('(x[1]+0.5*L)/L', '0'), L=L, degree=1)
v = interpolate(v, U)
load_test = assemble(inner(sigma(u),grad(v)) * dx)
print(load_test)
