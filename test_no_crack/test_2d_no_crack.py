# coding: utf-8
from dolfin import *
from DEM_cracking.DEM import *
from DEM_cracking.miscellaneous import *
from DEM_cracking.cracking import *
from scipy.sparse.linalg import cg

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# elastic parameters
rho = Constant(1180.)
nu = 0.35
E = 3.09e9
mu = E / (2.0*(1.0 + nu))
lambda_ = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))
penalty = 2*mu
Delta_u = 6.4e-3 #0.05e-3 #value from article

Ll, l0, H = 32e-3, 4.e-3, 16e-3
size_ref = 5 #5 #10 #20 #1 for debug
mesh = RectangleMesh(Point(0, H/2), Point(Ll, -H/2), size_ref*8, size_ref*4, "crossed")
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
h = mesh.hmax() #H / size_ref

# Sub domain for BC
def top(x, on_boundary):
    return near(x[1], H/2) and on_boundary

def down(x, on_boundary):
    return near(x[1], -H/2) and on_boundary

def right(x, on_boundary):
    return near(x[0], Ll) and on_boundary

def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

bnd_facets.set_all(0)
traction_boundary_1 = AutoSubDomain(top)
traction_boundary_1.mark(bnd_facets, 41)
traction_boundary_2 = AutoSubDomain(down)
traction_boundary_2.mark(bnd_facets, 42)
neumann_boundary_1 = AutoSubDomain(right)
neumann_boundary_1.mark(bnd_facets, 45)
neumann_boundary_2 = AutoSubDomain(left)
neumann_boundary_2.mark(bnd_facets, 47)
ds = Measure('ds')(subdomain_data=bnd_facets)

# Define BC
U_CR = VectorFunctionSpace(mesh, 'CR', 1)
hF = FacetArea(mesh)
v_CR = TestFunction(U_CR)

#new for BC
l4 = v_CR('+')[1] / hF * (ds(41) + ds(42))
#l4 = inner(v_CR('+'), as_vector((1.,1.))) / hF * ds
#l4 = inner(v_CR('+'), as_vector((1.,1.))) / hF * (ds(41)+ds(42))
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Creating the DEM problem
problem = DEMProblem(mesh, 2, penalty, nz_vec_BC, mu, lambda_)

#Imposing strongly Dirichlet BC
mat_not_D,mat_D = schur_matrices(problem)

def eps(v):
    return sym(v)

def sigma(eps_el):
    return lambda_ * tr(eps_el) * Identity(2) + 2.*mu * eps_el

#Variational problem
ref_elastic = ref_elastic_bilinear_form(problem, sigma, eps)
mat_elas = problem.elastic_bilinear_form(ref_elastic)

#Stresses output
problem.mat_stress = output_stress(problem, sigma, eps)

#useful
solution_u_DG = Function(problem.DG_0,  name="disp DG")
solution_stress = Function(problem.W, name="Stress")

#reference solution
x = SpatialCoordinate(mesh)
#Dirichlet BC
u_D = Expression(('0.', 'x[1]/fabs(x[1]) * disp'), disp = Delta_u,degree=1)

#paraview outputs
file = File('test.pvd')

#lhs
A = mat_elas + problem.mat_pen

#Removing Dirichlet BC
A_not_D = mat_not_D * A * mat_not_D.T
#rhs
rhs = -mat_not_D * A * problem.trace_matrix.T * np.nan_to_num(interpolate(u_D, U_CR).vector().get_local())

#inverting system
#u_reduced = spsolve(A_not_D, rhs)
u_reduced,info = cg(A_not_D, rhs)
assert(info == 0)

u = mat_not_D.T * u_reduced + problem.trace_matrix.T * np.nan_to_num(interpolate(u_D, U_CR).vector().get_local())

#sys.exit()

#Post-processing
vec_u_CR = problem.DEM_to_CR * u
vec_u_DG = problem.DEM_to_DG * u

#output initial conditions
solution_u_DG.vector().set_local(vec_u_DG)
solution_u_DG.vector().apply("insert")
file.write(solution_u_DG, 0)
solution_stress.vector().set_local(problem.mat_stress * problem.mat_grad * vec_u_CR)
solution_stress.vector().apply("insert")
file.write(solution_stress, 0)

print('Intial elastic energy: %.5e' % (0.5 * np.dot(u, A * u)))

#computation over
print('End of computation !')

