# coding: utf-8
import sys
sys.path.append('../..')
from facets import *
from scipy.sparse.linalg import eigsh,cg

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# elastic parameters
rho = Constant(1180.)
nu = 0.35
E = 3.09e9
mu = Constant(E / (2.0*(1.0 + nu)))
lambda_ = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))
penalty = 2.*float(mu)
Gc = 300
Delta_u = 0.05e-3 #value from article

Ll, l0, H = 32e-3, 4.e-3, 16e-3
size_ref = 1 #5 #10 #20 #1 for debug
mesh = RectangleMesh(Point(0, H/2), Point(Ll, -H/2), size_ref*8, size_ref*4, "crossed")
#mesh = RectangleMesh(Point(0, H/2), Point(Ll, -H/2), size_ref*8, size_ref*4, "left")
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
h = mesh.hmax() #H / size_ref
c = np.sqrt(float(mu/rho))

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

# Mesh-related functions
vol = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (vol('+') + vol('-'))/ (2. * hF('+'))
n = FacetNormal(mesh)

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour déplacement dans cellules
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
W = TensorFunctionSpace(mesh, 'DG', 0)
aux_CR = FunctionSpace(mesh, 'CR', 1) #Pour critère élastique. Reste en FunctionSpace
W_aux = VectorFunctionSpace(mesh, 'CR', 1)

#useful
for_dim = Function(U_DG)
dim = for_dim.geometric_dimension()
d = dim
solution_u_DG = Function(U_DG,  name="disp DG")
solution_v_DG = Function(U_DG,  name="vel DG")
solution_stress = Function(W, name="Stress")

#reference solution
x = SpatialCoordinate(mesh)
#quasi-ref solution
#Dirichlet BC
u_D = Expression(('0.', 'x[1]/fabs(x[1]) * disp'), disp = Delta_u,degree=2)

def eps(v):
    return sym(v)

def sigma(eps_el):
    return lambda_ * tr(eps_el) * Identity(2) + 2.*mu * eps_el

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
u_DG = TrialFunction(U_DG)
v_DG = TestFunction(U_DG)
Du_DG = TrialFunction(W)
Dv_DG = TestFunction(W)

#new for BC
l4 = v_CR('+')[1] / hF * (ds(41) + ds(42))
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)
print(nz_vec_BC)

#Cell-centre Galerkin reconstruction
nb_ddl_cells = U_DG.dofmap().global_dimension()
print('nb cell dof : %i' % nb_ddl_cells)
nb_ddl_ccG = nb_ddl_cells
facet_num = new_facet_neighborhood(mesh)
nb_ddl_CR = U_CR.dofmap().global_dimension()
initial_nb_ddl_CR = nb_ddl_CR #will not change. Useful for reducing u_CR for cracking criterion
nb_facet_original = nb_ddl_CR // d
print('nb dof CR: %i' % nb_ddl_CR)
G = connectivity_graph(mesh, d, penalty, nz_vec_BC)
print('ok graph !')
nb_ddl_ccG = nb_ddl_cells + len(nz_vec_BC)
print('nb ddl ccG: %i' % nb_ddl_ccG)
coord_bary,coord_num = smallest_convexe_bary_coord(mesh,facet_num,d,G) #, vertex_boundary)
print('Convexe ok !')
#matrice gradient
mat_grad = gradient_matrix(mesh, d)
print('gradient matrix ok !')
passage_ccG_to_CR,trace_matrix = matrice_passage_ccG_CR(mesh, coord_num, coord_bary, d, G, nb_ddl_ccG)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)
print('matrices passage ok !')
nb_ddl_grad = W.dofmap().global_dimension()

#write a test of P1-consistency of reconstruction etc...
#ones = np.ones(nb_ddl_cells)
func = SpatialCoordinate(mesh)
file = File('interpolation/test.pvd')
test = Function(U_CR)
u_ccG = np.zeros(nb_ddl_ccG)
u_ccG[:nb_ddl_cells] = local_project(func, U_DG).vector().get_local()
u_ccG += trace_matrix.T * local_project(func,U_CR).vector().get_local()
test.vector().set_local(passage_ccG_to_CR * u_ccG)
file.write(test)
file.write(local_project(grad(test), W))

##test of boundary facet reconstructions
#for (x,y) in G.edges():
#    if not G[x][y]['breakable']: #boundary facet
#        dof_CR = G[x][y]['dof_CR']
#        print(test.vector().get_local()[dof_CR])

