# coding: utf-8
import sys
sys.path.append('../')
from facets_no_vertex import *
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
G,nb_ddl_ccG_old = connectivity_graph(mesh, d, penalty)
print('nb dof ccG : %i' % nb_ddl_ccG_old)
nb_ddl_ccG = nb_ddl_ccG_old #nb de dof ccG avant fissuration
print('ok graph !')
coord_bary,coord_num = smallest_convexe_bary_coord(facet_num,dim,d,G,symmetric=True)
print('Convexe ok !')
#matrice gradient
mat_grad = gradient_matrix(mesh, d)
print('gradient matrix ok !')
passage_ccG_to_CR,matrice_trace_bord = matrice_passage_ccG_CR(mesh, nb_ddl_ccG, coord_num, coord_bary, d, G)
passage_ccG_to_DG = matrice_passage_ccG_DG(nb_ddl_cells,nb_ddl_ccG)
print('matrices passage ok !')
nb_ddl_grad = W.dofmap().global_dimension()

crack = graph_crack(mesh,dim,G,nb_ddl_cells,d) #to know the position of the crack more easily
cell_vertex = cells_containing_vertex(mesh, dim)
facet_vertex = facets_containing_vertex(mesh, dim)
vertex_facet = vertices_contained_in_facets(mesh, dim)
pos_vertex = pos_vertices(mesh, dim)
vertex_boundary = vertices_boundary(mesh)

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

#disp jump on facets
a49 = dot( jump(u_DG), avg(v_CR) ) / hF('+') * dS
A49 = assemble(a49)
row,col,val = as_backend_type(A49).mat().getValuesCSR()
disp_jump = sp.csr_matrix((val, col, row))

#average facet stresses
a50 = dot( dot( avg(sigma(eps(Du_DG))), n('-')), v_CR('-')) / hF('-') * dS + dot( dot( sigma(eps(Du_DG)), n), v_CR) / hF * ds
A50 = assemble(a50)
row,col,val = as_backend_type(A50).mat().getValuesCSR()
average_stresses = sp.csr_matrix((val, col, row))

#average facet strains
a51 = dot( dot( avg(eps(Du_DG)), n('-')), v_CR('-')) / hF('-') * dS + dot( dot( sigma(eps(Du_DG)), n), v_CR) / hF * ds
A51 = assemble(a51)
row,col,val = as_backend_type(A51).mat().getValuesCSR()
average_strains = sp.csr_matrix((val, col, row))

#penalty vector on each facet
a333 =  penalty / h_avg * inner(v_CR('+'), jump(u_DG)) * dS
A333 = assemble(a333)
row,col,val = as_backend_type(A333).mat().getValuesCSR()
mat_pen_facet = sp.csr_matrix((val, col, row))

#for outputs:
#Strain output
a2 = inner(eps(Du_DG), Dv_DG) / vol * dx
A2 = assemble(a2)
row,col,val = as_backend_type(A2).mat().getValuesCSR()
mat_strain = sp.csr_matrix((val, col, row))

#Stress output
a47 = inner(sigma(eps(Du_DG)), Dv_DG) / vol * dx
A47 = assemble(a47)
row,col,val = as_backend_type(A47).mat().getValuesCSR()
mat_stress = sp.csr_matrix((val, col, row))

##new for BC
#a4 = v_CR('+')[1] / hF * (ds(41) + ds(42)) #mode 1
##a4 = inner(v_CR('+'),as_vector((1.,1.))) / hF * (ds(41) + ds(42)) #mode mixte
##a4 = v_CR('+')[0] / hF * (ds(41) + ds(42)) #mode 2
#A4 = assemble(a4)
#A_BC = matrice_trace_bord.T * A4.get_local()
#nz = A_BC.nonzero()[0]
##A_BC[nz[0]+1] = 1. #pour bloquer déplacement solide rigide
#A_BC[nz[0]-1] = 1. #pour bloquer déplacement solide rigide

##Homogeneous Neumann BC
#L = np.zeros(nb_ddl_CR)

#Non homogeneous Neumann BC
sigma = 1.e3
#tau = 1.e-5 #sigma/3 #1.e3
zeta = 0 #1/8 #1/4 #1/2 #2/3 #3/4
tau = zeta * sigma / (1. - zeta)
x = SpatialCoordinate(mesh)
normal_stress_BC = Expression(('x[1]/fabs(x[1]) * tau * (1. + 1e-4*t)', 'x[1]/fabs(x[1]) * sigma * (1 + 1e-4*t)'), sigma=sigma, tau=tau, t=0, degree=2)
L = interpolate(normal_stress_BC, U_CR).vector().get_local()


file = File('mixed_new/test_%i_.pvd' % size_ref) #51)

count_output_energy_release = 0
count_output_disp = 0

count_output_crack = 0
cracked_facet_vertices = []
#cracked_facet_vertices_bis = []
cracked_facets = set()
broken_vertices = set()

#initial conditions
u = np.zeros(nb_ddl_ccG)

cracking_facets = set()
#before the computation begins, we break the facets
for (x,y) in G.edges():
    f = G[x][y]['dof_CR'][0] // d
    pos = G[x][y]['barycentre']
    if G[x][y]['breakable'] and pos[0] < l0 and np.absolute(pos[1]) < 1.e-10:
        cracking_facets.add(f)
        cracked_facet_vertices.append(G[x][y]['vertices']) #position of vertices of the broken facet
        crack.node[f]['broken'] = True #for graph of the crack

#adapting after crack
passage_ccG_to_DG, passage_ccG_to_CR, matrice_trace_bord, mat_grad, mat_not_D, mat_D, nb_ddl_CR, nb_ddl_ccG, facet_num, M_lumped, u, v, old_nb_dof_ccG = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, matrice_trace_bord, passage_ccG_to_DG, mat_grad, G, u, nb_ddl_ccG_old)
out_cracked_facets('mixed_new', size_ref, 0, cracked_facet_vertices, dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets
mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
mat_pen,mat_jump_1,mat_jump_2 = penalty_term(nb_ddl_ccG, mesh, d, dim, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR)
A = mat_elas + mat_pen
L = np.concatenate((L, np.zeros(d * len(cracking_facets))))
#Homogeneous Neumann BC on new crack lips
for f in cracked_facets:
    c = facet_num.get(f)[0]
    dof_CR = G[c][nb_ddl_cells // d + f]['dof_CR']
    L[dof_CR] = np.zeros(d)

##filling-in broken vertices
#for n in cracked_facets:
#    for nn in cracked_facets:
#        if crack.has_edge(n,nn):
#           broken_edges.add(crack[n][nn]['num'])

#filling-in broken vertices
for f in cracked_facets:
    for vert in vertex_facet.get(f):
        broken_vertices.add(vert)

#computing boundary of the crack
bnd_crack = boundary_crack_new(cracked_facets,broken_vertices,facet_vertex,vertex_boundary)
print(bnd_crack)

#definition of time-stepping parameters
chi = 4.5
dt = h / chi
print('dt: %.5e' % dt)
#sys.exit()
T = 1. #100 * dt
t = 0.
#normal_stress_BC.t -= dt #test

##intial values displacement and velocity
#v_not_D = mat_not_D * v
#u_not_D = mat_not_D * u

#Début boucle temporelle
while normal_stress_BC.t < T:
    normal_stress_BC.t += dt
    print('\n')
    print('t: %.5e' % normal_stress_BC.t)

    #Assembling rhs
    L[:initial_nb_ddl_CR] = interpolate(normal_stress_BC, U_CR).vector().get_local()
    for f in cracked_facets:
        c = facet_num.get(f)[0]
        dof_CR = G[c][nb_ddl_cells // d + f]['dof_CR']
        L[dof_CR] = np.zeros(d)
    L_aux = matrice_trace_bord.T * L

    #Computing new displacement values for non-Dirichlet dofs
    #u,info = cg(A,L_aux)
    #assert(info == 0)
    u = spsolve(A,L_aux)

    #Post-processing
    vec_u_CR = passage_ccG_to_CR * u
    vec_u_DG = passage_ccG_to_DG * u
    stresses = mat_stress * mat_grad * vec_u_CR
    stresses_per_cell = stresses.reshape((nb_ddl_cells // d,dim,d)) #For vectorial case
    strain = mat_strain * mat_grad * vec_u_CR
    strain_per_cell = strain.reshape((nb_ddl_cells // d,dim,d)) #For vectorial case

    ##outputs sometimes
    #if u_D.t % (T / 10) < dt:
    #solution_u_DG.vector().set_local(vec_u_DG)
    #solution_u_DG.vector().apply("insert")
    #file.write(solution_u_DG, normal_stress_BC.t)
    #solution_stress.vector().set_local(stresses)
    #solution_stress.vector().apply("insert")
    #file.write(solution_stress, normal_stress_BC.t)
    #sys.exit()
        
    #print('Elastic energy: %.5e' % (0.5*np.dot(u,mat_elas*u)))
    #print('Penalty energy: %.5e' % (0.5*np.dot(u,mat_pen*u)))
    #sys.exit()
    #stress_per_facet = average_stresses * mat_grad * vec_u_CR + mat_pen_facet * vec_u_DG
    #stress_per_facet = stress_per_facet.reshape((initial_nb_ddl_CR // d,d)) #For vectorial case


#    #test of cracking
#    cracking_facets = set()
#    energy_release_rates = dict()
#    biggest_G = -1.
#    vec_G = np.zeros(nb_ddl_cells // d)
#    vec_cell = np.zeros(nb_ddl_cells // d)
#    for tt,uu in bnd_crack.items():
#    #for tt in cracked_facets:   
#        #cell containing the broken facet
#        n1 = facet_num.get(tt)[0]
#        #pos_facet = G[n1][tt + nb_ddl_cells // d]['barycentre']
#        #print('Pos facet: (%f,%f)' % (pos_facet[0], pos_facet[1]))
#        for nn in crack.neighbors(tt):
#            if crack[tt][nn]['num'] == uu[0]: #right vertex
#                pos_vertex = crack[tt][nn]['barycentre']
#                break
#
#        #Computing discrete energy release rate
#        G_max = -1.
#        corresponding_cell = -1
#        #K1_save = 0.
#        #K2_save = 0.
#        for c in cell_vertex.get(uu[0]):
#            bary_cell = G.node[c]['pos']
#            #print('bary cell : (%f,%f)' % (bary_cell[0],bary_cell[1]))
#            dist = norm(bary_cell - pos_vertex)
#
#            #normal to the boundary
#            crack_normal = G[n1][tt + nb_ddl_cells // d]['normal']
#            #tangent to the boundary
#            crack_tangent = np.array([crack_normal[1], -crack_normal[0]])
#
#            #Stress for computing the SIF
#            dof = G.node[c]['dof']
#            average = stresses_per_cell[dof[0] // d,:]
#
#            #computing SIF
#            cell_stress = np.dot(average, crack_normal) #normal stress
#            K1 = np.dot(cell_stress, crack_normal) * np.sqrt(2.*np.pi*dist)
#            K2 = np.dot(cell_stress, crack_tangent) * np.sqrt(2.*np.pi*dist)
#            G_discrete = (1-nu*nu) / E * (K1*K1 + K2*K2) #Plane strain ?
#            biggest_G = max(biggest_G, G_discrete)
#            #print('K1: %.5e  K2: %.5e' % (K1,K2))
#            #print('G_h: %.5e' % G_discrete)
#            #density = np.tensordot(stresses_per_cell[c], strain_per_cell[c])
#            #vec_cell[c] = density
#            #print('Density elastic energy in cell: %.5e' % density)
#            #print('Facet energy: %.5e' % energy_density_per_facet[nn])
#
#            #Write selection of the breaking facet base on on the G_mas criterion per boundary edge...
#            if G_discrete > G_max:
#                corresponding_cell = c
#                G_max = G_discrete
#                #K1_save = K1
#                #K2_save = K2
#        print('G_max: %.5e' % biggest_G) #G_max)
#        #sys.exit()
#        if G_max >= Gc: #adding the facet from which their is a propagation in the dict
#            #energy_release_rates[tt] = [K1_save,K2_save] #G_max
#            #bary_cell = G.node[corresponding_cell]['pos']
#            #print('bary cell : (%f,%f)' % (bary_cell[0],bary_cell[1]))
#
#            #computing facet that shall break (if a facet shall break...)
#            max_density = -1
#            corresponding_facet = -1
#            for nn in crack.neighbors(tt):
#                if len(facet_num.get(nn)) == 2 and crack[tt][nn]['num'] == uu[0]: #edge of the crack tip is the right one
#                    c1,c2 = facet_num.get(nn)
#                    pos_facet = G[c1][c2]['barycentre']
#                    print('Pos facet: (%f,%f)' % (pos_facet[0], pos_facet[1]))
#                    r = norm(pos_facet - pos_vertex)
#                    
#                    #computing density
#                    dof_1 = G.node[c1]['dof']
#                    dof_2 = G.node[c2]['dof']
#                    facet_stress = 0.5 * (stresses_per_cell[dof_1[0] // d,:] + stresses_per_cell[dof_2[0] // d,:])
#                    facet_strain = (1+nu)/E * facet_stress - nu/E * np.trace(facet_stress) * np.eye(dim)
#                    density = np.tensordot(r * facet_stress, facet_strain)
#                    print('Density elastic energy in facet: %.5e' % density)
#                    if density > max_density:
#                        max_density = density
#                        corresponding_facet = nn
#                        
#            #Adding chosen facet to the list of breaking facets
#            cracking_facets.add(corresponding_facet)
#            broken_edges.add(crack[corresponding_facet][tt]['num'])
#            #print(corresponding_facet)
#            n1,n2 = facet_num.get(corresponding_facet)
#            pos = G[n1][n2]['barycentre']
#            cracked_facet_vertices.append(G[n1][n2]['vertices']) #position of vertices of the broken facet
#            print('Barycentre cracking facet: (%f,%f)' % (pos[0],pos[1]))


    #test of cracking
    cracking_facets = set()
    energy_release_rates = dict()
    biggest_G = -1.
    vec_G = np.zeros(nb_ddl_cells // d)
    vec_cell = np.zeros(nb_ddl_cells // d)
    for vert in bnd_crack:
        G_max = -1.
        pos_v = pos_vertex.get(vert)
        for f in facet_vertex.get(vert):
            if f in cracked_facets: #It should be a facet of the crack that contains the vertex
                assert(len(facet_num.get(f)) == 1) #boundary facet
                cell_crack = facet_num.get(f)[0]
                #facet related quantities
                crack_normal = G[cell_crack][f + nb_ddl_cells // d]['normal']
                crack_tangent = np.array([-crack_normal[1], crack_normal[0]])
                #loop on cells containing the vertex to compute SIF
                for c in cell_vertex.get(vert):
                    pos_cell = G.node[c]['pos']
                    dist = np.linalg.norm(pos_v - pos_cell)
                    dof_cell = G.node[c]['dof']
                    stress = stresses_per_cell[dof_cell[0] // d,:]
                    normal_stress = np.dot(stress, crack_normal)

                    #SIF computation #plane stress
                    K1 = np.dot(normal_stress, crack_normal) * np.sqrt(2.*np.pi*dist) #normale à la facette tant que fissure reste droite...
                    #K1 = 0 if K1 < 0 else K1 #checking that crack is not in compression
                    G1 = K1 * K1 / E #plane stress
                    K2 = np.dot(normal_stress, crack_tangent) * np.sqrt(2.*np.pi*dist) #normale à la facette tant que fissure reste droite...
                    G2 = K2 * K2 / E #plane stress
                    G_discrete = G1 + G2

                    #Gmax criterion
                    if G_discrete > G_max:
                        G_max = G_discrete

        #testing if vertex propagates
        print(G_max)
        if G_max >= Gc:
            #kinking criterion
            max_density = -1
            corresponding_facet = -1
            values_density = dict()
            for f in facet_vertex.get(vert):
                #print(facet_num.get(f))
                if len(facet_num.get(f)) == 2: #not breaking a boundary facet
                    c1,c2 = facet_num.get(f)
                    pos_facet = G[c1][c2]['barycentre']
                    #print('Pos facet: (%f,%f)' % (pos_facet[0], pos_facet[1]))
                    r = np.linalg.norm(pos_facet - pos_v)
                    
                    #computing density
                    dof_1 = G.node[c1]['dof']
                    dof_2 = G.node[c2]['dof']
                    facet_stress = 0.5 * (stresses_per_cell[dof_1[0] // d,:] + stresses_per_cell[dof_2[0] // d,:])
                    facet_strain = 0.5 * (strain_per_cell[dof_1[0] // d,:] + strain_per_cell[dof_2[0] // d,:])
                    density = np.tensordot(r * facet_stress, facet_strain)
                    #print('Density elastic energy in facet: %.5e' % density)
                    values_density[f] = density
            #Getting the breaking facets
            max_density = max(list(values_density.values()))
            threshold = 1.e-5 #on ne prend que le max...
            for f,density in values_density.items():
                if np.absolute(max_density - density) < threshold * np.absolute(max_density):
                    cracking_facets.add(f) #Adding chosen facet to the list of breaking facets
                    #adding vertices of the facet in the set of broken vertices
                    for ff in crack.neighbors(f):
                        broken_vertices.add(crack[f][ff]['num'])
                    #print(corresponding_facet)
                    n1,n2 = facet_num.get(f)
                    cracked_facet_vertices.append(G[n1][n2]['vertices']) #position of vertices of the broken facet

    if len(cracking_facets) > 0:
        #storing the number of ccG dof before adding the new facet dofs
        old_nb_dof_ccG = nb_ddl_ccG

        #outputs
        solution_u_DG.vector().set_local(vec_u_DG)
        solution_u_DG.vector().apply("insert")
        file.write(solution_u_DG, normal_stress_BC.t)
        solution_stress.vector().set_local(stresses)
        solution_stress.vector().apply("insert")
        file.write(solution_stress, normal_stress_BC.t)
        #sys.exit() #just for test

        count_output_crack +=1

        #adapting after crack
        #penalty terms modification
        mat_jump_1_aux,mat_jump_2_aux = removing_penalty(mesh, d, dim, nb_ddl_ccG, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, cracking_facets, facet_num)
        mat_jump_1 -= mat_jump_1_aux
        mat_jump_2 -= mat_jump_2_aux
        #modifying mass and rigidity matrix beacuse of breaking facets...
        passage_ccG_to_DG, passage_ccG_to_CR, matrice_trace_bord, mat_grad, mat_not_D, mat_D, nb_ddl_CR, nb_ddl_ccG, facet_num, M_lumped, u, v, old_nb_dof_ccG = adapting_after_crack(cracking_facets, cracked_facets, d, dim, facet_num, nb_ddl_cells, nb_ddl_ccG, nb_ddl_CR, passage_ccG_to_CR, matrice_trace_bord, passage_ccG_to_DG, mat_grad, G, u, nb_ddl_ccG_old)
        out_cracked_facets('mixed_new',size_ref, count_output_crack, cracked_facet_vertices, dim) #paraview cracked facet file

        #assembling new rigidity matrix and mass matrix after cracking
        mat_elas = elastic_term(mat_grad, passage_ccG_to_CR)
        #penalty_terms modification
        mat_jump_1_aux,mat_jump_2_aux = adding_boundary_penalty(mesh, d, dim, nb_ddl_ccG, mat_grad, passage_ccG_to_CR, G, nb_ddl_CR, cracking_facets, facet_num)
        mat_jump_1.resize((nb_ddl_CR,nb_ddl_ccG))
        mat_jump_2.resize((nb_ddl_CR,nb_ddl_grad))
        mat_jump_1 += mat_jump_1_aux
        mat_jump_2 += mat_jump_2_aux
        mat_jump = mat_jump_1 + mat_jump_2 * mat_grad * passage_ccG_to_CR
        mat_pen = mat_jump.T * mat_jump
        A = mat_elas + mat_pen #n'a pas de termes pour certains dof qu'on vient de créer
        L = np.concatenate((L, np.zeros(d * len(cracking_facets)))) #because we have homogeneous Neumann BC on the crack lips !
        #for f in cracked_facets.union(cracking_facets):
        #    c = facet_num.get(f)[0]
        #    dof_CR = G[c][nb_ddl_cells // d + f]['dof_CR']
        #    L[dof_CR] = np.zeros(d)

    cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

    bnd_crack = boundary_crack_new(cracked_facets,broken_vertices,facet_vertex,vertex_boundary)
    print(bnd_crack)
    assert(len(bnd_crack) > 0)

print('End of computation !')
