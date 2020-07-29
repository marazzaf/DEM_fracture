# coding: utf-8
import sys
sys.path.append('../')
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
penalty = 2*float(mu)
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

#useful
for_dim = Function(U_DG)
dim = for_dim.geometric_dimension()
d = dim
solution_u_DG = Function(U_DG,  name="disp DG")
solution_v_DG = Function(U_DG,  name="vel DG")
solution_stress = Function(W, name="Stress")

#reference solution
x = SpatialCoordinate(mesh)
#Dirichlet BC
u_D = Expression(('0.', 'x[1]/fabs(x[1]) * disp'), disp = Delta_u,degree=2)
sigma_normal = E*n

def eps(v):
    return sym(v)

def sigma(eps_el):
    return lambda_ * tr(eps_el) * Identity(2) + 2.*mu * eps_el

#Cell-centre Galerkin reconstruction
nb_ddl_cells = U_DG.dofmap().global_dimension()
print('nb cell dof : %i' % nb_ddl_cells)
nb_ddl_ccG = nb_ddl_cells
facet_num = new_facet_neighborhood(mesh)
nb_ddl_CR = U_CR.dofmap().global_dimension()
initial_nb_ddl_CR = nb_ddl_CR #will not change. Useful for reducing u_CR for cracking criterion
nb_facet_original = nb_ddl_CR // d
print('nb dof CR: %i' % nb_ddl_CR)
G = connectivity_graph(mesh, d, penalty)
print('ok graph !')
coord_bary,coord_num = smallest_convexe_bary_coord(mesh,facet_num,d,G) #, vertex_boundary)
print('Convexe ok !')
#matrice gradient
mat_grad = gradient_matrix(mesh, d)
print('gradient matrix ok !')
passage_ccG_to_CR = matrice_passage_ccG_CR(mesh, coord_num, coord_bary, d, G)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells)
print('matrices passage ok !')
nb_ddl_grad = W.dofmap().global_dimension()

##write a test of P1-consistency of reconstruction etc...
##ones = np.ones(nb_ddl_cells)
#func = SpatialCoordinate(mesh)
#file = File('interpolation/test.pvd')
#test = Function(U_CR)
#test.vector().set_local(passage_ccG_to_CR * local_project(func, U_DG).vector().get_local())
#file.write(test)
#file.write(local_project(grad(test), W))
#sys.exit()

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
u_DG = TrialFunction(U_DG)
v_DG = TestFunction(U_DG)
Du_DG = TrialFunction(W)
Dv_DG = TestFunction(W)

#Variational problem
a1 = inner(sigma(eps(Du_DG)), Dv_DG) * dx #does not change with topological changes
A1 = assemble(a1)
row,col,val = as_backend_type(A1).mat().getValuesCSR()
A1 = sp.csr_matrix((val, col, row))

def elastic_term(mat_grad_, passage):
    return  passage.T * mat_grad_.T * A1 * mat_grad_ * passage

#Stress output
a47 = inner(sigma(eps(Du_DG)), Dv_DG) / vol * dx
A47 = assemble(a47)
row,col,val = as_backend_type(A47).mat().getValuesCSR()
mat_stress = sp.csr_matrix((val, col, row))

#new for BC
l4 = v_CR('+')[1] / hF * (ds(41) + ds(42))
#l4 = v_CR('+')[1] / hF * ds
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
#nz_vec_BC.append(nz_vec_BC[0]-1) #fixing rigid body movement
nz_vec_BC = set(nz_vec_BC)
#for (x,y) in G.edges():
#    dof_CR = set(G[x][y]['dof_CR'])
#    if len(dof_CR & nz_vec_BC) > 0:
#        print(G[x][y]['barycentre'])
##print(len(nz_vec_BC))
#print(nz_vec_BC)
#sys.exit()

#special for full Neumann BC
nz_vec_BC = set()

#Corresponding penalty term for rhs
#def penalty_force(passage_CR): #Nitsche penalty to impose weakly BC
#    a4 = penalty * hF / vol * v_CR[1] * u_D[1] * (ds(41) + ds(42))
#    A4 = assemble(a4).get_local()
#    A4.resize(passage_CR.shape[0])
#    return passage_CR.T * A4 #goes into the forces...

#WRITE A NEW PENALTY TERM THAT WILL PENALIZE VALUE AT FACET BARYCENTRE !!!!

def penalty_force(passage_CR): #Nitsche penalty to impose weakly BC
    aux = penalty * hF / vol * v_CR[1] * (ds(41) + ds(42))
    vec_aux = assemble(aux).get_local()
    A4 = np.zeros(nb_ddl_CR)
    for x,y in G.edges():
        f = G[x][y]['num']
        pos_facet = G[x][y]['barycentre']
        dof_CR = G[x][y]['dof_CR']
        with_BC = list(set(dof_CR) & nz_vec_BC)
        if len(with_BC) > 0: #A Dirichelt BC is imposed on facet
            for i in with_BC:
                A4[i] = vec_aux[i] * u_D(pos_facet)[1]

    return passage_CR.T * A4 #goes into the forces...

#Homogeneous Neumann BC
L = np.zeros(nb_ddl_CR)

#paraview outputs
file = File('test_neumann/test.pvd')

count_output_crack = 0
cracked_facet_vertices = []
broken_vertices = set()
last_broken_vertices = set()
cracked_facets = set()

#initial conditions
u = np.zeros(nb_ddl_ccG)
v = np.zeros(nb_ddl_ccG)

cracking_facets = set()
##before the computation begins, we break the facets to have a crack of length 1
#for (x,y) in G.edges():
#    f = G[x][y]['dof_CR'][0] // d
#    pos = G[x][y]['barycentre']
#    if G[x][y]['breakable'] and np.abs(pos[1]) < 1.e-15 and pos[0] < l0:
#        cracking_facets.add(f)
#        #crack.node[f]['broken'] = True
#        cracked_facet_vertices.append(G[x][y]['vertices']) #position of vertices of the broken facet
#        
#passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G)
#out_cracked_facets('test', size_ref, 0, cracked_facet_vertices, dim) #paraview cracked facet file
#count_output_crack +=1
#cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets
mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
mat_pen,mat_jump_1,mat_jump_2 = penalty_term(nb_ddl_ccG, mesh, d, dim, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, nz_vec_BC)
A = mat_elas + mat_pen
L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

#taking into account exterior loads and penalty for Dirichlet BC
#rhs = passage_ccG_to_CR.T * L + penalty_force(passage_ccG_to_CR)
#rhs = interpolate(Constant((0.,1.)), U_DG).vector().get_local()
#rhs = penalty_force(passage_ccG_to_CR)

#Full Neumann BC
#rhs = passage_ccG_to_CR.T * assemble(dot(sigma_normal, v_CR('+')) * ds).get_local()
#rhs = passage_ccG_to_CR.T * assemble(dot(sigma_normal, v_CR('+')) * (ds(41) + ds(42))).get_local()
rhs = passage_ccG_to_CR.T * assemble(sigma_normal[1] * v_CR('+')[1] * (ds(41) + ds(42))).get_local()
#rhs_aux = assemble(sigma_normal[1] * v_CR('+')[1] * (ds(41) + ds(42))).get_local()
#print(rhs_aux.nonzero())
print(rhs.nonzero())
#sys.exit()

#inverting system
#u = spsolve(A, rhs)
u,info = cg(A, rhs)
assert(info == 0)

#Post-processing
vec_u_CR = passage_ccG_to_CR * u
vec_u_DG = passage_ccG_to_DG * u

#output initial conditions
solution_u_DG.vector().set_local(vec_u_DG)
solution_u_DG.vector().apply("insert")
file.write(solution_u_DG, 0)
test = Function(U_CR)
test.vector().set_local(vec_u_CR[:initial_nb_ddl_CR])
file.write(test)
file.write(local_project(grad(test), W))
solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
solution_stress.vector().apply("insert")
file.write(solution_stress, 0)

print('Intial elastic energy: %.5e' % (0.5 * np.dot(u, A * u)))

#computation over
print('End of computation !')

