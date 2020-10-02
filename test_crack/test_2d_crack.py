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
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Creating the DEM problem
problem = DEMProblem(mesh, 2, penalty, nz_vec_BC, mu, lambda_)

def eps(v):
    return sym(v)

def sigma(eps_el):
    return lambda_ * tr(eps_el) * Identity(2) + 2.*mu * eps_el

#Variational problem
ref_elastic = ref_elastic_bilinear_form(problem, sigma, eps)

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

count_output_crack = 1
cracked_facet_vertices = []
cracking_facets = set()
cracked_facets = set()

#before the computation begins, we break the facets to have a crack of length 1
for (x,y) in problem.Graph.edges():
    f = problem.Graph[x][y]['dof_CR'][0] // problem.d
    pos = problem.Graph[x][y]['barycentre']
    if problem.Graph[x][y]['breakable'] and abs(pos[1]) < 1.e-15 and pos[0] < l0:
        cracking_facets.add(f)
        cracked_facet_vertices.append(problem.Graph[x][y]['vertices']) #position of vertices of the broken facet

#adapting after crack
mat_jump_1_aux,mat_jump_2_aux = problem.removing_penalty(cracking_facets)
problem.mat_jump_1 -= mat_jump_1_aux
problem.mat_jump_2 -= mat_jump_2_aux
problem.adapting_after_crack(cracking_facets, cracked_facets)
folder='./'
out_cracked_facets(folder, size_ref, 0, cracked_facet_vertices, problem.dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets
mat_elas = problem.elastic_bilinear_form(ref_elastic)
problem.mat_jump_1.resize((problem.nb_dof_CR,problem.nb_dof_DEM))
problem.mat_jump_2.resize((problem.nb_dof_CR,problem.nb_dof_grad))
problem.mat_jump = problem.mat_jump_1 + problem.mat_jump_2 * problem.mat_grad * problem.DEM_to_CR
mat_pen = problem.mat_jump.T * problem.mat_jump
A = mat_elas + mat_pen

#lhs
A = mat_elas + mat_pen

#Removing Dirichlet BC
A_not_D = problem.mat_not_D * A * problem.mat_not_D.T
#rhs
rhs = -problem.mat_not_D * A * problem.trace_matrix.T * np.nan_to_num(interpolate(u_D, U_CR).vector().get_local())

#inverting system
#u_reduced = spsolve(A_not_D, rhs)
u_reduced,info = cg(A_not_D, rhs)
assert(info == 0)

u = problem.complete_solution(u_reduced, u_D)

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

#computation over
print('End of computation !')

