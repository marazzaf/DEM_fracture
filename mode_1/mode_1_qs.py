# coding: utf-8
import sys
sys.path.append('../')
from facets import *
from scipy.sparse.linalg import cg,spsolve

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
# elastic parameters
E = 3.09e9 
nu = 0.35 
mu    = Constant(E / (2.0*(1.0 + nu)))
lambda_ = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))
penalty = float(mu)
Gc = 300
k = 1. #loading speed

#sample dimensions
Ll, l0, H = 32e-3, 4e-3, 16e-3

#mesh
size_ref = 5 #20 #10 #5 #1 #debug
mesh = RectangleMesh(Point(0., H/2), Point(Ll, -H/2), size_ref*8, size_ref*4, "crossed")
folder = 'structured'
#size_ref = 0
#mesh = Mesh('mesh/plate_1_E_4.xml')
h = mesh.hmax()
#finir plus tard pour taille des mailles.
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

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

# Mesh-related functions
vol = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (vol('+') + vol('-'))/ (2. * hF('+'))
n = FacetNormal(mesh)

#Function spaces
U_DG = VectorFunctionSpace(mesh, 'DG', 0) #Pour délacement dans cellules
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
W = TensorFunctionSpace(mesh, 'DG', 0)
aux_CR = FunctionSpace(mesh, 'CR', 1) #Pour critère élastique. Reste en FunctionSpace
W_aux = VectorFunctionSpace(mesh, 'CR', 1) #Exceptionnel pour  cas vectoriel

#useful
for_dim = Function(U_DG)
dim = for_dim.geometric_dimension()
d = dim #vectorial problem
solution_u_DG = Function(U_DG,  name="disp DG")
solution_stress = Function(W, name="Stress")

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
u_DG = TrialFunction(U_DG)
v_DG = TestFunction(U_DG)
Du_DG = TrialFunction(W)
Dv_DG = TestFunction(W)

#new for BC
l4 = v_CR('+')[1] / hF * (ds(41) + ds(42)) #mode 1
#a4 = inner(v_CR('+'),as_vector((1.,1.))) / hF * (ds(41) + ds(42)) #mode mixte
#a4 = v_CR('+')[0] / hF * (ds(41) + ds(42)) #mode 2
L4 = assemble(l4)
vec_BC = L4.get_local()
nz = vec_BC.nonzero()
#vec_BC[nz[0]+1] = 1. #pour bloquer déplacement solide rigide
vec_BC[nz[0]-1] = 1. #pour bloquer déplacement solide rigide
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

##Homogeneous Neumann BC
#L = np.zeros(nb_ddl_CR)

##Non homogeneous Neumann BC
#sigma = 1.e3
##tau = 1.e-5 #sigma/3 #1.e3
#zeta = 0 #1/8 #1/4 #1/2 #2/3 #3/4
#tau = zeta * sigma / (1. - zeta)
#x = SpatialCoordinate(mesh)
#normal_stress_BC = Expression(('x[1]/fabs(x[1]) * tau * (1. + 1e-4*t)', 'x[1]/fabs(x[1]) * sigma * (1 + 1e-4*t)'), sigma=sigma, tau=tau, t=0, degree=2)
#L = interpolate(normal_stress_BC, U_CR).vector().get_local()

#Load and non-homogeneous Dirichlet BC
def eps(v): #v is a gradient matrix
    return sym(v)

def sigma(eps_el):
    return lambda_ * tr(eps_el) * Identity(dim) + 2.*mu * eps_el

#Cell-centre Galerkin reconstruction
nb_ddl_cells = U_DG.dofmap().global_dimension()
print('nb cell dof : %i' % nb_ddl_cells)
facet_num = new_facet_neighborhood(mesh)
nb_ddl_CR = U_CR.dofmap().global_dimension()
initial_nb_ddl_CR = nb_ddl_CR #will not change. Useful for reducing u_CR for cracking criterion
nb_facet_original = nb_ddl_CR // d
print('nb dof CR: %i' % nb_ddl_CR)
G = connectivity_graph(mesh, d, penalty, nz_vec_BC)
print('ok graph !')
nb_ddl_ccG = nb_ddl_cells + len(nz_vec_BC)
coord_bary,coord_num = smallest_convexe_bary_coord(mesh,facet_num,d,G)
print('Convexe ok !')
#matrice gradient
mat_grad = gradient_matrix(mesh, d)
print('gradient matrix ok !')
passage_ccG_to_CR,trace_matrix = matrice_passage_ccG_CR(mesh, coord_num, coord_bary, d, G, nb_ddl_ccG)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)
passage_ccG_to_DG_1,ccG_to_DG_1_aux_1,ccG_to_DG_1_aux_2 = matrice_passage_ccG_DG_1(mesh, nb_ddl_ccG, d, dim, mat_grad, passage_ccG_to_CR)
facet_to_facet = linked_facets(mesh,dim,G) #designed to lighten research of potentialy failing facets close to a broken facet
facets_cell = facets_in_cell(mesh,d)
nb_ddl_grad = W.dofmap().global_dimension()
mat_not_D,mat_D = schur(nb_ddl_cells, nb_ddl_ccG)

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

#average facet stresses
a50 = dot( dot( avg(sigma(eps(Du_DG))), n('-')), v_CR('-')) / hF('-') * dS #n('-')
A50 = assemble(a50)
row,col,val = as_backend_type(A50).mat().getValuesCSR()
average_stresses = sp.csr_matrix((val, col, row))

#facet areas
f_CR = TestFunction(aux_CR)
areas = assemble(f_CR('+') * (dS + ds)).get_local() #bien écrit !

##Homogeneous Neumann BC
#L = np.zeros(nb_ddl_CR)

##Non homogeneous Neumann BC
#sigma = 1.e3
##tau = 1.e-5 #sigma/3 #1.e3
#zeta = 0 #1/8 #1/4 #1/2 #2/3 #3/4
#tau = zeta * sigma / (1. - zeta)
#x = SpatialCoordinate(mesh)
#normal_stress_BC = Expression(('x[1]/fabs(x[1]) * tau * (1. + 1e-4*t)', 'x[1]/fabs(x[1]) * sigma * (1 + 1e-4*t)'), sigma=sigma, tau=tau, t=0, degree=2)
#L = interpolate(normal_stress_BC, U_CR).vector().get_local()


file = File('structured/test_%i_.pvd' % size_ref) #51)

count_output_energy_release = 0
count_output_disp = 0

count_output_crack = 0
cracked_facet_vertices = []
cracked_facets = set()
broken_vertices = set()

#initial conditions
u = np.zeros(nb_ddl_ccG)

cracking_facets = set()
cells_to_test = set()
#before the computation begins, we break the facets
for (x,y) in G.edges():
    f = G[x][y]['dof_CR'][0] // d
    pos = G[x][y]['barycentre']
    if G[x][y]['breakable'] and pos[0] < l0 and np.absolute(pos[1]) < 1.e-10:
        cracking_facets.add(f)
        cracked_facet_vertices.append(G[x][y]['vertices']) #position of vertices of the broken facet
        cells_to_test |= set(facet_num.get(f)) #verifying only one facet per cell breaks

#adapting after crack
passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)
out_cracked_facets(folder, size_ref, 0, cracked_facet_vertices, dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets
mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
mat_pen,mat_jump_1,mat_jump_2 = penalty_term(nb_ddl_ccG, mesh, d, dim, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR)
A = mat_elas + mat_pen
L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

##Homogeneous Neumann BC on new crack lips
#for f in cracked_facets:
#    c = facet_num.get(f)[0]
#    dof_CR = G[c][nb_ddl_cells // d + f]['dof_CR']
#    L[dof_CR] = np.zeros(d)

#definition of time-stepping parameters
chi = 4.5
dt = h / chi
print('dt: %.5e' % dt)
T = 1. #100 * dt
t = 0.

#Début boucle temporelle
#while normal_stress_BC.t < T:
#    normal_stress_BC.t += dt
#    print('\n')
#    print('t: %.5e' % normal_stress_BC.t)
#
#    #Assembling rhs
#    L[:initial_nb_ddl_CR] = interpolate(normal_stress_BC, U_CR).vector().get_local()
#    for f in cracked_facets:
#        c = facet_num.get(f)[0]
#        dof_CR = G[c][nb_ddl_cells // d + f]['dof_CR']
#        L[dof_CR] = np.zeros(d)
#    L_aux = matrice_trace_bord.T * L

while u_D.t < T:
    u_D.t += dt
    print('BC disp: %.5e' % (k * u_D.t))
    inverting = True

    #interpolation of Dirichlet BC
    FF = interpolate(u_D, U_CR).vector().get_local()
    F = mat_D * trace_matrix.T * FF

    #Assembling rhs
    L_not_D = -B * F

    count = 0
    while inverting:
        #inverting system
        count += 1
        print('COUNT: %i' % count)
        u_reduced,info = cg(A_not_D, L_not_D)
        assert(info == 0)
        u = mat_not_D.T * u_reduced + mat_D.T * F

        #Post-processing
        vec_u_CR = passage_ccG_to_CR * u
        vec_u_DG = passage_ccG_to_DG * u
        facet_stresses = average_stresses * mat_grad * vec_u_CR

        ##outputs sometimes
        #if u_D.t % (T / 10) < dt:
        #solution_u_DG.vector().set_local(vec_u_DG)
        #solution_u_DG.vector().apply("insert")
        #file.write(solution_u_DG, normal_stress_BC.t)
        #solution_stress.vector().set_local(stresses)
        #solution_stress.vector().apply("insert")
        #file.write(solution_stress, normal_stress_BC.t)
        #sys.exit()

        #Cracking criterion
        stress_per_facet = average_stresses * mat_grad * vec_u_CR #plain stress
        Gh = stress_per_facet * stress_per_facet
        Gh *= 0.5 * np.pi / mu * areas
        #removing energy of already cracked facets
        Gh[list(cracked_facets)] = np.zeros(len(cracked_facets))

        #breaking one facet at a time
        args = np.argpartition(Gh, -20)[-20:] #is 20 enough?
        for f in args: 
            assert f not in cracked_facets
            if Gh[f] > Gc and f in potentially_cracking_facets: #otherwise not cracking !
                #Verifying that it is the only facet of two cells to break
                c1,c2 = facet_num.get(f)
                test_1 = cracked_facets & facets_cell.get(c1)
                test_2 = cracked_facets & facets_cell.get(c2)
                if len(test_1) == 0 and len(test_2) == 0:
                    cracking_facets = {f}
                    print(Gh[f])
                    #print(G[c1][c2]['barycentre'])
                    cracked_facet_vertices.append(G[c1][c2]['vertices']) #position of vertices of the broken facet
                    potentially_cracking_facets |= facet_to_facet.get(f) #updating set
                    cells_to_test |= set(facet_num.get(f))
                    break #When we get a facet verifying the conditions, we stop the search and continue with the cracking process
            else:
                inverting = False

        if len(cracking_facets) > 0:
            #outputs
            solution_u_DG.vector().set_local(vec_u_DG)
            solution_u_DG.vector().apply("insert")
            file.write(solution_u_DG, normal_stress_BC.t)
            solution_stress.vector().set_local(stresses)
            solution_stress.vector().apply("insert")
            file.write(solution_stress, normal_stress_BC.t)

        #get correspondance between dof
        for f in cracking_facets:
            #sys.exit()
            n1,n2 = facet_num.get(f)
            cracked_facet_vertices.append(G[n1][n2]['vertices']) #position of vertices of the broken facet
            pos = G[n1][n2]['barycentre']
            #print('dof num: %i' % f)
            print('pos bary facet : (%f,%f)' % (pos[0], pos[1]))
            potentially_cracking_facets -= (facets_cell.get(n1) | facets_cell.get(n2)) #so that no other facet of these two cells will break

        #to be sure not to break facets of a cell that already has a broken facet
        for f in cracked_facets:
            c1 = facet_num.get(f)[0]
            potentially_cracking_facets -= facets_cell.get(c1)

        #treatment if the crack propagates
        if len(cracking_facets) > 0:
            count_output_crack +=1

        #adapting after crack
        #penalty terms modification
        mat_jump_1_aux,mat_jump_2_aux = removing_penalty(mesh, d, dim, nb_ddl_ccG, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, cracking_facets, facet_num)
        mat_jump_1 -= mat_jump_1_aux
        mat_jump_2 -= mat_jump_2_aux
        #modifying mass and rigidity matrix beacuse of breaking facets...
        passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)
        out_cracked_facets(folder,size_ref, count_output_crack, cracked_facet_vertices, dim) #paraview cracked facet file

        #assembling new rigidity matrix and mass matrix after cracking
        mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
        passage_ccG_to_DG_1 = ccG_to_DG_1_aux_1 + ccG_to_DG_1_aux_2 * mat_grad * passage_ccG_to_CR #recomputed
        #penalty_terms modification
        mat_jump_1.resize((nb_ddl_CR,nb_ddl_ccG))
        mat_jump_2.resize((nb_ddl_CR,nb_ddl_grad))
        mat_jump = mat_jump_1 + mat_jump_2 * mat_grad * passage_ccG_to_CR
        mat_pen = mat_jump.T * mat_jump
        A = mat_elas + mat_pen #n'a pas de termes pour certains dof qu'on vient de créer
        L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

        #Imposing strongly Dirichlet BC
        A_D = mat_D * A * mat_D.T
        A_not_D = mat_not_D * A * mat_not_D.T
        B = mat_not_D * A * mat_D.T
        F = mat_D * trace_matrix.T * FF
        L_not_D = -B * F

    cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

print('End of computation !')
