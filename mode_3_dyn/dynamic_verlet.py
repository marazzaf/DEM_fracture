# coding: utf-8
import sys
sys.path.append('../..')
from facets import *
from scipy.sparse.linalg import eigsh
from ufl import sign

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
#parameters['linear_algebra_backend'] = 'PETSc'

# elastic parameters
rho = 1.
mu = .5
penalty = mu
Gc = 0.01 #1.5e-2 ???
cs = np.sqrt(mu / rho) #shear wave velocity
k = .15 #loading speed...

Ll, l0, H = 6., 1., 1.
size_ref = 10 #40 #20 #10
mesh = RectangleMesh(Point(0, H), Point(Ll, -H), size_ref*6, 2*size_ref, "crossed")
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
h = H / size_ref #(5*size_ref)
#print('Shear wave velocity: %.5e' % cs)
#print('dt ref: %.5e' % (h/cs))

# Sub domain for BC
def upper_left(x, on_boundary):
    return near(x[0], 0.) and x[1] >= 0. and on_boundary

def lower_left(x, on_boundary):
    return near(x[0], 0.) and x[1] <= 0. and on_boundary

def right(x, on_boundary):
    return near(x[0], Ll) and on_boundary

def top_down(x, on_boundary):
    return near(np.absolute(x[1]), H/2) and on_boundary

bnd_facets.set_all(0)
traction_boundary_1 = AutoSubDomain(upper_left)
traction_boundary_1.mark(bnd_facets, 41)
traction_boundary_2 = AutoSubDomain(lower_left)
traction_boundary_2.mark(bnd_facets, 42)
clamped_boundary = AutoSubDomain(right)
clamped_boundary.mark(bnd_facets, 45)
neumann_boundary = AutoSubDomain(top_down)
neumann_boundary.mark(bnd_facets, 47)
ds = Measure('ds')(subdomain_data=bnd_facets)

# Mesh-related functions
vol = CellVolume(mesh) #Pour volume des particules voisines
hF = FacetArea(mesh)
h_avg = (vol('+') + vol('-'))/ (2. * hF('+'))
n = FacetNormal(mesh)

#Function spaces
U_DG = FunctionSpace(mesh, 'DG', 0) #Pour déplacement dans cellules
U_CR = FunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
W = VectorFunctionSpace(mesh, 'DG', 0) #, shape=(2,2))

#useful
for_dim = Function(U_DG)
dim = for_dim.geometric_dimension()
d = 1 #scalar problem
solution_u_DG = Function(U_DG,  name="disp DG")
solution_v_DG = Function(U_DG,  name="vel DG")
solution_stress = Function(W, name="Stress")

#definition of time-stepping parameters
#T = 1.2 / k
#T = 2. / k
#T = 3. / k
T = 4./k
tl = T / 20

#reference solution
x = SpatialCoordinate(mesh)
#quasi-ref solution
#Dirichlet BC
g0 = Expression('t <= tl ? 0.5*k * t * t / tl * (1 - x[0]/L) : k * t - 0.5*k*tl', L=Ll, tl=tl, k=k, t=0, degree=2)
u_D = g0 * sign(x[1])
h0 = Expression('t <= tl ? k * t / tl * (1 - x[0]/L) : k', L=Ll, tl=tl, k=k, t=0, degree=2)
v_D = h0 * sign(x[1])

#Load and non-homogeneous Dirichlet BC
def eps(v): #v is a gradient matrix
    return v

def sigma(eps_el):
    return mu * eps_el #mu

# Define variational problem
u_CR = TrialFunction(U_CR)
v_CR = TestFunction(U_CR)
u_DG = TrialFunction(U_DG)
v_DG = TestFunction(U_DG)
Du_DG = TrialFunction(W)
Dv_DG = TestFunction(W)

#new for BC
l4 = v_CR('+') / hF * (ds(41) + ds(42) + ds(45))
L4 = assemble(l4)
vec_BC = L4.get_local()
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Cell-centred Galerkin reconstruction
nb_ddl_cells = U_DG.dofmap().global_dimension()
print('nb cell dof : %i' % nb_ddl_cells)
facet_num = new_facet_neighborhood(mesh)
nb_ddl_CR = U_CR.dofmap().global_dimension()
initial_nb_ddl_CR = nb_ddl_CR #will not change. Useful for reducing u_CR for cracking criterion
print('nb dof CR: %i' % nb_ddl_CR)
G = connectivity_graph(mesh, d, penalty, nz_vec_BC)
nb_ddl_ccG = nb_ddl_cells + len(nz_vec_BC)
print('ok graph !')
coord_bary,coord_num = smallest_convexe_bary_coord(mesh,facet_num,d,G)
print('Convexe ok !')
#matrice gradient
mat_grad = gradient_matrix(mesh, d)
print('gradient matrix ok !')
passage_ccG_to_CR,trace_matrix = matrice_passage_ccG_CR(mesh, coord_num, coord_bary, d, G, nb_ddl_ccG)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)
passage_ccG_to_DG_1,ccG_to_DG_1_aux_1,ccG_to_DG_1_aux_2 = matrice_passage_ccG_DG_1(mesh, nb_ddl_ccG, d, dim, mat_grad, passage_ccG_to_CR)
facet_to_facet = linked_facets(mesh,dim,G) #designed to lighten research of potentialy failing facets close to a broken facet
nb_ddl_grad = W.dofmap().global_dimension()
mat_not_D,mat_D = schur(nb_ddl_cells, nb_ddl_ccG)
print('matrices passage Schur ok')

#Variational problem
a1 = inner(sigma(eps(Du_DG)), Dv_DG) * dx #does not change with topological changes
A1 = assemble(a1)
row,col,val = as_backend_type(A1).mat().getValuesCSR()
A1 = sp.csr_matrix((val, col, row))

def elastic_term(mat_grad_, passage):
    return  passage.T * mat_grad_.T * A1 * mat_grad_ * passage

#average facet stresses
a50 = dot( dot( avg(sigma(eps(Du_DG))), n('-')), v_CR('-')) / hF('-') * dS #n('-')
A50 = assemble(a50)
row,col,val = as_backend_type(A50).mat().getValuesCSR()
average_stresses = sp.csr_matrix((val, col, row))

#for outputs:
#Stress output
a47 = inner(sigma(eps(Du_DG)), Dv_DG) / vol * dx
A47 = assemble(a47)
row,col,val = as_backend_type(A47).mat().getValuesCSR()
mat_stress = sp.csr_matrix((val, col, row))

#Assembling mass matrix
M_lumped = mass_matrix(mesh, d, dim, rho, nb_ddl_ccG)  # M_lumped est un vecteur
print('Mass matrix assembled !')

#Homogeneous Neumann BC
L = np.zeros(nb_ddl_CR)

#paraview outputs
#file = File('k_0_1/antiplane_%i_.pvd' % size_ref)

#length crack output
length_crack = open('other_speeds/k_2_c/length_crack_%i.txt' % size_ref, 'w')

f_CR = TestFunction(U_CR)
areas = assemble(f_CR('+') * (dS + ds)).get_local() #bien écrit !

count_output_crack = 0
cracked_facets_vertices = []
cracked_facets = set()
length_cracked_facets = 0.

#initial conditions
u = np.zeros(nb_ddl_ccG)
v = np.zeros(nb_ddl_ccG)

cracking_facets = set()

#assembling rigidity matrix
mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
mat_pen,mat_jump_1,mat_jump_2 = penalty_term(nb_ddl_ccG, mesh, d, dim, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, nz_vec_BC)
passage_ccG_to_DG_1 = ccG_to_DG_1_aux_1 + ccG_to_DG_1_aux_2 * mat_grad * passage_ccG_to_CR #recomputed
A = mat_elas + mat_pen
L = np.concatenate((L, np.zeros(d * len(cracking_facets))))

#for Verlet integration
v_old = np.zeros_like(v)

#Imposing strongly Dirichlet BC
A_D = mat_D * A * mat_D.T
A_not_D = mat_not_D * A * mat_not_D.T
B = mat_not_D * A * mat_D.T
M_not_D = mat_not_D * M_lumped

#sorties paraview avant début calcul
#file.write(solution_u_DG, 0)

# Time-stepping parameters
eig_M = min(M_not_D)
eig_K = eigsh(A_not_D, k=1, return_eigenvectors=False) #which='LM' (largest in magnitude, great)
#eig_K = max(np.real(eigenvalues_K))
dt = np.sqrt(eig_M / eig_K)
print('dt: %.5e' % dt)
print('chi: %.5f' % (h / (k*dt)))

u_not_D = np.zeros(nb_ddl_cells)
v_not_D = np.zeros_like(nb_ddl_cells)

#Début boucle temporelle
while u_D.t < T:
    u_D.t += dt
    print('BC disp: %.5e' % (k * u_D.t))
    v_D.t += dt

    #Computing new displacement values for non-Dirichlet dofs
    u_not_D = u_not_D + dt * v_not_D

    #interpolation of Dirichlet BC
    #displacement
    FF = interpolate(u_D, U_CR).vector().get_local()
    F = mat_D * trace_matrix.T * FF

    #computing new disp values
    u = mat_not_D.T * u_not_D + mat_D.T * F

    #Post-processing
    vec_u_CR = passage_ccG_to_CR * u
    vec_u_DG = passage_ccG_to_DG * u

    ##sorties paraview
    #if u_D.t % (T / 12) < dt:
    #    solution_u_DG.vector().set_local(vec_u_DG)
    #    solution_u_DG.vector().apply("insert")
    #    file.write(solution_u_DG, u_D.t)
    #    solution_v_DG.vector().set_local(passage_ccG_to_DG * v)
    #    solution_v_DG.vector().apply("insert")
    #    file.write(solution_v_DG, u_D.t)
    #    solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
    #    solution_stress.vector().apply("insert")
    #    file.write(solution_stress, u_D.t)

    #Test with SIF
    n1 = facet_num.get(closest)[0]
    #n2 = facet_num.get(-closest)[0]
    pos_closest = G[n1][nb_ddl_cells // d + closest]['barycentre'][0]
    #print(pos_closest)
    for f in facet_to_facet.get(closest):
        if len(facet_num.get(f)) == 2:
            n1,n2 = facet_num.get(f)
            pos = G[n1][n2]['barycentre']
            #print(pos)
            if pos[0] > pos_closest and np.absolute(pos[1]) < 1.e-15:
                facet_stresses = average_stresses * mat_grad * vec_u_CR
                dof_CR = G[n1][n2]['dof_CR']
                #SIF = facet_stresses[dof_CR] * np.sqrt(np.pi * h) #scalar problem
                SIF = facet_stresses[dof_CR] * np.sqrt(np.pi * areas[f])
                G_3 = 0.5 * SIF * SIF / mu #for model mu
                break

    #Il faut trouver la facette qui se trouve devant celle qu'on regarde...
    #adding a criterion on the position of the crack
    cracking_facets = set()
    for f in facet_to_facet.get(closest): #facets linked sharing a set of codimension 2 with last broken facet
    #for (x,y) in G.edges(): #all facets !
        if len(facet_num.get(f)) == 2: #one of the facet may already have broken
            x,y = facet_num.get(f)
            pos = G[x][y]['barycentre']
            if G_3 >= Gc and np.absolute(pos[1]) < 1.e-13: #si facet est au milieu du maillage... #and len(facet_num.get(f)) == 2
                cracking_facets.add(f)
                closest = f #for next potential break...
                break #on ne casse les facet que une par une pour l'instant

    ##getting cracking facets
    #cracking_facets_aux = set( (energy_release_rate_per_facet - Gc > 1.e-5 * Gc ).nonzero()[0] ) #gives the number of the (scalar) dof of the facets that break !
    #cracking_facets_aux.difference_update(cracked_facets) #not breaking again already broken facets
    ##adding a criterion on the position of the crack
    #cracking_facets = set()
    #for f in cracking_facets_aux:
    #    n1,n2 = facet_num.get(f) #On casse une face de bord ? Etrange...
    #    if np.abs(G[n1][n2]['barycentre'][1]) < 1.e-10: #si facet est au milieu du maillage...
    #        cracking_facets.add(f)

    ##for computation of crack speed
    #if len(cracking_facets) > 0:
    #    old_length_crack = length_cracked_facets
    #    last_t_crack = u_D.t

    #get correspondance between dof
    for f in cracking_facets:
        n1,n2 = facet_num.get(f)
        cracked_facets_vertices.append(G[n1][n2]['vertices']) #position of vertices of the broken facet
        pos = G[n1][n2]['barycentre']
        print('dof num: %i' % f)
        print('pos bary facet : (%f,%f)' % (pos[0], pos[1]))
        length_cracked_facets += areas[f]

    #treatment if the crack propagates
    if len(cracking_facets) > 0:
        ##output
        #solution_u_DG.vector().set_local(vec_u_DG)
        #solution_u_DG.vector().apply("insert")
        #file.write(solution_u_DG, u_D.t)
        #solution_v_DG.vector().set_local(passage_ccG_to_DG * v)
        #solution_v_DG.vector().apply("insert")
        #file.write(solution_v_DG, u_D.t)
        #solution_stress.vector().set_local(mat_stress * mat_grad * vec_u_CR)
        #solution_stress.vector().apply("insert")
        #file.write(solution_stress, u_D.t)

        #output with length of the crack in 2d
        length_crack.write('%.5e %.5e\n' % (k * u_D.t * H, length_cracked_facets))
        
        #storing the number of ccG dof before adding the new facet dofs
        old_nb_dof_ccG = nb_ddl_ccG

        #out_cracked_facets('k_0_2', size_ref, count_output_crack, cracked_facets_vertices, dim) #paraview cracked facet file
        count_output_crack +=1

        ##output with length of the crack in 2d
        #length_crack.write('%.5e %.5e\n' % (k * u_D.t * H, length_cracked_facets))

        #adapting after crack
        #penalty terms modification
        mat_jump_1_aux,mat_jump_2_aux = removing_penalty(mesh, d, dim, nb_ddl_ccG, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, cracking_facets, facet_num)
        mat_jump_1 -= mat_jump_1_aux
        mat_jump_2 -= mat_jump_2_aux
        passage_ccG_to_CR, mat_grad, nb_ddl_CR, facet_num, mat_D, mat_not_D = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, mat_grad, G, mat_D, mat_not_D)

        #assembling new rigidity matrix and mass matrix after cracking
        passage_ccG_to_DG_1 = ccG_to_DG_1_aux_1 + ccG_to_DG_1_aux_2 * mat_grad * passage_ccG_to_CR #recomputed
        mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
        #penalty_terms modification
        #mat_jump_1_aux,mat_jump_2_aux = adding_boundary_penalty(mesh, d, dim, nb_ddl_ccG, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, cracking_facets, facet_num)
        mat_jump_1.resize((nb_ddl_CR,nb_ddl_ccG))
        mat_jump_2.resize((nb_ddl_CR,nb_ddl_grad))
        #mat_jump_1 += mat_jump_1_aux
        #mat_jump_2 += mat_jump_2_aux
        mat_jump = mat_jump_1 + mat_jump_2 * mat_grad * passage_ccG_to_CR
        mat_pen = mat_jump.T * mat_jump
        A = mat_elas + mat_pen #n'a pas de termes pour certains dof qu'on vient de créer
        L = np.concatenate((L, np.zeros(d * len(cracking_facets)))) #because we have homogeneous Neumann BC on the crack lips !

        #Imposing strongly Dirichlet BC
        A_D = mat_D * A * mat_D.T
        A_not_D = mat_not_D * A * mat_not_D.T
        B = mat_not_D * A * mat_D.T
        #M_not_D = mat_not_D * M_lumped #not changing

        #recomputing time-step to adapt to the new rigidity matrix and the new mass matrix
        dt_old = dt
        #eig_M = min(M_not_D)
        eig_K = eigsh(A_not_D, k=1, return_eigenvectors=False) #which='LM' (largest in magnitude)
        dt = np.sqrt(eig_M / eig_K)
        if abs(dt_old - dt) / dt_old > 0.01:
            print('Time integration not stable any longer. Terminating computation')
            sys.exit()
        #print('dt: %.5e' % dt)

        ##resize pour ajouter valeurs des nouveaux dofs
        #vec_u_CR_aux = vec_u_CR.copy()
        #vec_u_CR_aux = vec_u_CR_aux.reshape((vec_u_CR_aux.size // d, d))
        #u.resize(nb_ddl_ccG)
        #v.resize(nb_ddl_ccG)
        ##same modification on v_old because of Verlet integration
        #v_old.resize(nb_ddl_ccG, refcheck=False)
        #
        ##putting new values in the new facet dofs
        #for f in cracking_facets: #filling in per broken facet the reconstructed values
        #    new_dof_ccG_1 = G.node[nb_ddl_cells // d + f]['dof'] #to be filled with newly computed value
        #    assert(G.node[nb_ddl_cells // d + f]['sign'] > 0)
        #    new_dof_ccG_2 = G.node[-nb_ddl_cells // d - f]['dof'] #to be filled with newly computed value
        #    assert(G.node[-nb_ddl_cells // d - f]['sign'] < 0)
        #
        #    #valeurs new dofs disp
        #    previous_facet_value = vec_u_CR_aux[f,:]
        #    u[new_dof_ccG_1] = previous_facet_value
        #    u[new_dof_ccG_2] = previous_facet_value
        #
        #    #valeurs new dofs vel
        #    nei1 = nx.neighbors(G,nb_ddl_cells // d + f)
        #    nei2 = nx.neighbors(G,-nb_ddl_cells // d - f)
        #    nei1_aux = 0
        #    nei2_aux = 0
        #    for i,j in zip(nei1,nei2):
        #        nei1_aux = i
        #        nei2_aux = j
        #    v[new_dof_ccG_1] = v[nei1_aux]
        #    v[new_dof_ccG_2] = v[nei2_aux]
        #    #for v_old (Verlet integration)
        #    v_old[new_dof_ccG_1] = v_old[nei1_aux]
        #    v_old[new_dof_ccG_2] = v_old[nei2_aux]
        #    
        #u_not_D = mat_not_D * u
        #v_not_D = mat_not_D * v

    cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

    #Finishing updates after cracking
    #Calcul des forces.
    #integral = mat_not_D * trace_matrix.T * L - A_not_D * u_not_D - B * F
    integral = -A_not_D * u_not_D - B * F
    Res = integral / M_not_D

    #interpolation of Dirichlet BC
    #velocity
    vel_interpolate = interpolate(v_D, U_CR).vector().get_local()
    vel_D = mat_D * trace_matrix.T * vel_interpolate

    #Computing new velocities
    v_not_D = mat_not_D * v_old + Res * dt
    v = mat_not_D.T * v_not_D + mat_D.T * vel_D
    v_old = v

#computation over
length_crack.close()
print('End of computation !')

