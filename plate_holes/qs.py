# coding: utf-8
from dolfin import *
from scipy.sparse.linalg import cg
from DEM_cracking.DEM import *
from DEM_cracking.miscellaneous import *
from DEM_cracking.cracking import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# elastic parameters
E = 210e9 
nu = 0.3
mu    = Constant(E / (2.0*(1.0 + nu)))
lambda_ = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))
penalty = float(mu)
Gc = 2.7e-3
#k = 1.e-4 #loading speed

#sample dimensions
Ll, l0, H = 65e-3, 10e-3, 120e-3

#mesh
folder = 'results'
mesh = Mesh()
#size_ref = 2
#with XDMFFile("mesh/fine.xdmf") as infile:
#    infile.read(mesh)
size_ref = 1
with XDMFFile("mesh/coarse.xdmf") as infile:
    infile.read(mesh)
h = mesh.hmax()
#finir plus tard pour taille des mailles.
bnd_facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

d = 2 #vectorial case 

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
hF = FacetArea(mesh)

#Function spaces
U_CR = VectorFunctionSpace(mesh, 'CR', 1) #Pour interpollation dans les faces
v_CR = TestFunction(U_CR)

#new for BC
#l4 = v_CR('+')[1] / hF * (ds(41) + ds(42)) #mode 1
l4 = inner(v_CR('+'),as_vector((1.,1.))) / hF * (ds(41) + ds(42)) #mode mixte
#l4 = v_CR('+')[0] / hF * (ds(41) + ds(42)) #mode 2
L4 = assemble(l4)
vec_BC = L4.get_local()
nz = vec_BC.nonzero()
#vec_BC[nz[0]+1] = 1. #pour bloquer déplacement solide rigide
#vec_BC[nz[0]-1] = 1. #pour bloquer déplacement solide rigide
nz_vec_BC = list(vec_BC.nonzero()[0])
nz_vec_BC = set(nz_vec_BC)

#Creating the DEM problem
problem = DEMProblem(mesh, d, penalty, nz_vec_BC, mu)

#For Dirichlet BC
x = SpatialCoordinate(mesh)
u_D = Expression(('x[1] > 0 ? t : 0', '0'), t=0, degree=1)

#Load and non-homogeneous Dirichlet BC
def eps(v): #v is a gradient matrix
    return sym(v)

def sigma(eps_el):
    return lambda_ * tr(eps_el) * Identity(problem.dim) + 2.*mu * eps_el

#Variational problem
ref_elastic = ref_elastic_bilinear_form(problem, sigma, eps)
mat_elas = problem.elastic_bilinear_form(ref_elastic)

#Stresses output
problem.mat_stress = output_stress(problem, sigma, eps)
#Facet jump output
problem.mat_jump_bis = problem.mat_jump()

#useful
solution_u_DG = Function(problem.DG_0,  name="disp DG")
solution_stress = Function(problem.W, name="Stress")

#For outputs
file = File('%s/test_%i_.pvd' % (folder,size_ref))

count_output_crack = 1
cracked_facet_vertices = []
cracked_facets = set()
cells_with_cracked_facet = set()
not_breakable_facets = set()
cracking_facets = set()
broken_vertices = set()

#before the computation begins, we break the facets
for (x,y) in problem.Graph.edges():
    f = problem.Graph[x][y]['dof_CR'][0] // d
    pos = problem.Graph[x][y]['barycentre']
    if problem.Graph[x][y]['breakable'] and abs(pos[1] - (H/2-55e-3)) < 1.e-15 and pos[0] < l0:
        cracking_facets.add(f)
        cracked_facet_vertices.append(problem.Graph[x][y]['vertices']) #position of vertices of the broken facet
        cells_with_cracked_facet |= {x,y}
        broken_vertices |= set(problem.Graph[x][y]['vertices_ind'])

#adapting after crack
problem.removing_penalty(cracking_facets)
problem.adapting_after_crack(cracking_facets, cracked_facets)
problem.update_penalty_matrix()
problem.elastic_bilinear_form(ref_elastic)
A = problem.mat_elas + problem.mat_pen

#After modifications
out_cracked_facets(folder, size_ref, 0, cracked_facet_vertices, problem.dim) #paraview cracked facet file
cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

#Homogeneous Neumann BC
L = np.zeros(problem.nb_dof_CR)

#Updating facets that cannot be broken because they belong to a cell with an already broken facet
not_breakable_facets |= cracked_facets #already broken facets cannot break again
for c in cells_with_cracked_facet:
    not_breakable_facets |= problem.facets_cell.get(c)

#Imposing strongly Dirichlet BC
A_not_D,B = problem.schur_complement(A)

#definition of time-stepping parameters
chi = 1
dt = 1e-6 #1e-4 #weird...
print('dt: %.5e' % dt)
T = 0.02e-3
u_D.t = 0

while u_D.t < T:
    u_D.t += dt
    print('BC disp: %.5e' % (u_D.t))
    inverting = True

    #interpolation of Dirichlet BC
    FF = interpolate(u_D, U_CR).vector().get_local()
    F = problem.mat_D * problem.trace_matrix.T * FF

    #Assembling rhs
    L_not_D = -B * F

    count = 0
    while inverting:
        #inverting system
        count += 1
        print('COUNT: %i' % count)
        u_reduced,info = cg(A_not_D, L_not_D)
        assert(info == 0)
        u = problem.complete_solution(u_reduced,u_D)

        #Post-processing
        vec_u_CR = problem.DEM_to_CR * u
        vec_u_DG = problem.DEM_to_DG * u
        stresses = problem.mat_stress * problem.mat_grad * vec_u_CR
        stress_per_cell = stresses.reshape((problem.nb_dof_cells // problem.d,problem.dim,problem.dim))

        ##outputs sometimes
        #if u_D.t % (T / 10) < dt:
        #solution_u_DG.vector().set_local(vec_u_DG)
        #solution_u_DG.vector().apply("insert")
        #file.write(solution_u_DG, u_D.t)
        #solution_stress.vector().set_local(stresses)
        #solution_stress.vector().apply("insert")
        #file.write(solution_stress, u_D.t)
        #sys.exit()

        cracking_facets = set()

        ##Computing new Gh
        #Gh = problem.energy_release_rates(vec_u_CR, cracked_facets, not_breakable_facets)
        #
        ##Potentially cracking facet with biggest Gh
        #idx = np.argpartition(Gh, -20)[-20:] #is 20 enough?
        #indices = idx[np.argsort((-Gh)[idx])]
        #
        ##Choosing which facet to break
        #for f in indices:
        #    if Gh[f] > Gc:
        #        cracking_facets = {f}
        #        c1,c2 = problem.facet_num.get(f)
        #        cells_with_cracked_facet |= {c1,c2}
        #        not_breakable_facets |= (problem.facets_cell.get(c1) | problem.facets_cell.get(c2))
        #        broken_vertices |= set(problem.Graph[c1][c2]['vertices_ind'])
        #        break #When we get a facet verifying the conditions, we stop the search and continue with the cracking process
        #    else:
        #        inverting = False

        #Computing Gh per vertex and then kinking criterion
        Gh_v = problem.energy_release_rate_vertex_bis(broken_vertices, cracked_facets, vec_u_CR, vec_u_DG)

        #Looking for facet with largest Gh
        idx = np.argpartition(Gh_v, -20)[-20:] #is 20 enough?
        indices = idx[np.argsort((-Gh_v)[idx])]
        #print(indices)
        #print(Gh_v[1631])
        #print(problem.facets_vertex.get(1631))
        #print(cracked_facets)
        #print(set(problem.facets_vertex.get(1631)) & cracked_facets)
        #print('stop')

        #Kinking to choose breaking facet
        for v in indices:
            if Gh_v[v] > Gc:
                f = K2_kinking_criterion(problem, v, vec_u_CR, not_breakable_facets,cracked_facets)
                if f != None:
                    cracking_facets = {f}
                    c1,c2 = problem.facet_num.get(f)
                    cells_with_cracked_facet |= {c1,c2}
                    not_breakable_facets |= (problem.facets_cell.get(c1) | problem.facets_cell.get(c2))
                    broken_vertices |= set(problem.Graph[c1][c2]['vertices_ind'])
                    break #When we get a facet verifying the conditions, we stop the search and continue with the cracking process
        if len(cracking_facets) == 0:
            inverting = False

        if len(cracking_facets) > 0:
            #outputs
            solution_u_DG.vector().set_local(vec_u_DG)
            solution_u_DG.vector().apply("insert")
            file.write(solution_u_DG, u_D.t)
            solution_stress.vector().set_local(problem.mat_stress * problem.mat_grad * vec_u_CR)
            solution_stress.vector().apply("insert")
            file.write(solution_stress, u_D.t)

        #get correspondance between dof
        for f in cracking_facets:
            n1,n2 = problem.facet_num.get(f)
            cracked_facet_vertices.append(problem.Graph[n1][n2]['vertices']) #position of vertices of the broken facet
            pos = problem.Graph[n1][n2]['barycentre']
            print('pos bary facet : (%f,%f)' % (pos[0], pos[1]))

        #treatment if the crack propagates
        if len(cracking_facets) > 0:
            #adapting after crack
            problem.removing_penalty(cracking_facets)
            problem.adapting_after_crack(cracking_facets, cracked_facets)
            problem.update_penalty_matrix()
            problem.elastic_bilinear_form(ref_elastic)
            A = problem.mat_elas + problem.mat_pen
            L = np.concatenate((L, np.zeros(problem.d * len(cracking_facets))))
            
            #Crack output
            count_output_crack +=1
            out_cracked_facets(folder, size_ref, count_output_crack, cracked_facet_vertices, problem.dim) #paraview cracked facet file

            #Imposing strongly Dirichlet BC
            A_not_D,B = problem.schur_complement(A)
            F = problem.mat_D * problem.trace_matrix.T * FF
            L_not_D = -B * F

            cracked_facets.update(cracking_facets) #adding facets just cracked to broken facets

print('End of computation !')
