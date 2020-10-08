#coding: utf-8
from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np
from DEM_cracking.facet_reconstruction import bary_coord
from scipy.sparse import lil_matrix,dok_matrix
import networkx as nx

def adapting_after_crack(self, cracking_facets, already_cracked_facets):
    impacted_facets = self.adapting_graph(cracking_facets)
    self.adapting_facet_reconstruction(cracking_facets, already_cracked_facets, impacted_facets)
    self.adapting_grad_matrix(cracking_facets)
    self.DEM_to_DG_1 = self.aux_DG_1_1 + self.aux_DG_1_2 * self.mat_grad * self.DEM_to_CR
    return

def adapting_graph(self, cracking_facets):
    impacted_facets = set()
    #Modifying connectivity graph
    for f in cracking_facets:
        n1,n2 = self.facet_num.get(f) #n1 : plus, n2 : minus    
        #retrieving facets impacted by rutpure:
        impacted_facets.update(self.Graph[n1][n2]['recon']) #impacted facets will have a new CR reconstruction
        ex_dofs = self.Graph[n1][n2]['dof_CR']
        ex_num = self.Graph[n1][n2]['num']
        ex_bary = self.Graph[n1][n2]['barycentre']
        ex_normal = self.Graph[n1][n2]['normal']
        ex_measure = self.Graph[n1][n2]['measure']
        ex_vertices = self.Graph[n1][n2]['vertices']
        ex_vertices_ind = self.Graph[n1][n2]['vertices_ind']
        ex_pen = self.Graph[n1][n2]['pen_factor']
        
        #removing link between the two cell dofs
        self.Graph.remove_edge(n1,n2)
        
        #adding the new facet dofs
        self.Graph.add_node(self.nb_dof_cells // self.d + f, pos=ex_bary, sign=int(1), vertices=np.array([])) #linked with n1
        self.Graph.add_node(-self.nb_dof_cells // self.d - f, pos=ex_bary, sign=int(-1), vertices=np.array([])) #linked with n2

        #adding the connectivity between cell dofs and new facet dofs
        self.Graph.add_edge(n1, self.nb_dof_cells // self.d + f, num=ex_num, dof_CR=ex_dofs, measure=ex_measure, barycentre=ex_bary, normal=ex_normal, breakable=False, vertices=ex_vertices, pen_factor = ex_pen, vertices_ind = ex_vertices_ind)
        self.facet_num[f] = [n1]
        new_dof_CR = list(np.arange(self.nb_dof_CR, self.nb_dof_CR+self.d))
        self.Graph.add_edge(-self.nb_dof_cells // self.d - f, n2, num=self.nb_dof_CR//self.d, dof_CR=new_dof_CR, measure=ex_measure, barycentre=ex_bary, normal=ex_normal, breakable=False, vertices=ex_vertices, pen_factor = ex_pen, vertices_ind = ex_vertices_ind)
        self.nb_dof_CR += self.d
        self.facet_num[-f] = [n2]

    return impacted_facets #will be used to update CR reconstruction

def adapting_facet_reconstruction(self, cracking_facets, already_cracked_facets, impacted_facets):
    #Modifying matrix passage_CR
    self.DEM_to_CR.resize((self.nb_dof_CR,self.nb_dof_DEM))
    passage_CR_new = self.DEM_to_CR.tolil()

    tetra_coord_bary = dict()
    tetra_coord_num = dict()
    #fill rows for new dof
    #Putting ones in new facet dofs and putting 0 where there were barycentric reconstructions
    for f in cracking_facets:
        n1 = self.facet_num.get(f)[0]
        n2 = self.facet_num.get(-f)[0]
        num_global_ddl_1 = self.Graph[n1][f + self.nb_dof_cells // self.d]['dof_CR']
        num_global_ddl_2 = self.Graph[-f - self.nb_dof_cells // self.d][n2]['dof_CR']
            
        #deleting previous barycentric reconstruction
        passage_CR_new[num_global_ddl_1[0], :] = np.zeros(passage_CR_new.shape[1])
        if self.d >= 2: #deuxième ligne
            passage_CR_new[num_global_ddl_1[1], :] = np.zeros(passage_CR_new.shape[1])
        if self.d == 3: #troisième ligne
            passage_CR_new[num_global_ddl_1[2], :] = np.zeros(passage_CR_new.shape[1])
        #recomputing the CR reconstruction
        coord_bary_1,coord_num_1,connectivity_recon_1 = bary_coord(f, self)
        coord_bary_2,coord_num_2,connectivity_recon_2 = bary_coord(-f, self)

        #Filling-in the new reconstruction
        for i1,j1,i2,j2 in zip(coord_num_1,coord_bary_1,coord_num_2,coord_bary_2):
            passage_CR_new[num_global_ddl_1[0],i1[0]] += j1
            passage_CR_new[num_global_ddl_2[0],i2[0]] += j2 
            if self.d >= 2:
                passage_CR_new[num_global_ddl_1[1],i1[1]] += j1
                passage_CR_new[num_global_ddl_2[1],i2[1]] += j2
            if self.d == 3:
                passage_CR_new[num_global_ddl_1[2],i1[2]] += j1
                passage_CR_new[num_global_ddl_2[2],i2[2]] += j2

    impacted_facets.difference_update(already_cracked_facets) #removing already cracked facets from the set of facets that need a new CR reconstruction

    #The connectivity recon will be recomputed. Thus retrieving the num of the facet from the potentially impacted. Will be upadted afterwards.
    for (u,v,r) in self.Graph.edges(data='recon'):
        if r != None:
            r.difference_update(impacted_facets)

    #computing the new CR reconstruction for impacted facets
    for f in impacted_facets:
        #if len(face_num.get(g)) > 1: #facet not on boundary otherwise reconstruction does not change
        coord_bary,coord_num,connectivity_recon = bary_coord(f, self)
        tetra_coord_bary[f] = coord_bary
        tetra_coord_num[f] = coord_num

        #updating the connectivity_recon in the graph. Adding the new connectivity recon in graph.
        for k in connectivity_recon:
            if(len(self.facet_num.get(k))) == 2:
                n1,n2 = self.facet_num.get(k)
                self.Graph[n1][n2]['recon'].add(f)

    #Putting-in the new barycentric coordinates for inner facet reconstruction that changed
    for f in impacted_facets:
        if len(self.facet_num.get(f)) > 1: #facet not on boundary
            n1,n2 = self.facet_num.get(f)
            num_global_ddl = self.Graph[n1][n2]['dof_CR']
        else: #facet on boundary
            n = self.facet_num.get(f)[0]
            num_global_ddl = self.Graph[n][f + self.nb_dof_cells // self.d]['dof_CR']
        #erasing previous values
        passage_CR_new[num_global_ddl[0],:] = np.zeros(passage_CR_new.shape[1])
        if self.d >= 2:
            passage_CR_new[num_global_ddl[1],:] = np.zeros(passage_CR_new.shape[1])
        if self.d == 3:
            passage_CR_new[num_global_ddl[2],:] = np.zeros(passage_CR_new.shape[1])
            #Putting new values in
        for i,j in zip(tetra_coord_num.get(f),tetra_coord_bary.get(f)):
            passage_CR_new[num_global_ddl[0],i[0]] += j 
            if self.d >= 2:
                passage_CR_new[num_global_ddl[1],i[1]] += j
            if self.d == 3:
                passage_CR_new[num_global_ddl[2],i[2]] += j

    #Optimization
    self.DEM_to_CR = passage_CR_new.tocsr()
    self.DEM_to_CR.eliminate_zeros()

    return

def adapting_grad_matrix(self, cracking_facets):
    #Modifying the gradient reconstruction matrix
    #No renumbering because uses CR dofs and not ccG dofs !!!
    self.mat_grad.resize((self.mat_grad.shape[0],self.nb_dof_CR))
    mat_grad_new = self.mat_grad.tolil() #only changes for cells having a facet that broke
    for f in cracking_facets:
        #first newly facet dof and associated cell
        c1 = self.facet_num[f][0]
        mat_grad_new[c1 * self.d * self.dim : (c1+1) * self.d * self.dim, :] = local_gradient_matrix(c1, self)
        c2 = self.facet_num[-f][0]
        mat_grad_new[c2 * self.d * self.dim : (c2+1) * self.d * self.dim, :] = local_gradient_matrix(c2, self)

    self.mat_grad = mat_grad_new.tocsr()
    
    return

def out_cracked_facets(folder,num_computation, num_output, cracked_facets_vertices, dim_): #Sortie au format vtk
    crack = open('%s/crack_%i_%i.vtk' % (folder,num_computation,num_output), 'w')
    nb_facets = len(cracked_facets_vertices)

    #header of vtk file
    crack.write("# vtk DataFile Version 3.0\n")
    crack.write("#Simulation Euler\n")
    crack.write("ASCII\n")
    crack.write('\n') #saut de ligne
    crack.write("DATASET UNSTRUCTURED_GRID\n")
    crack.write("POINTS %i DOUBLE\n" % (dim_ * nb_facets))

    nb_vertex = 0
    for f in cracked_facets_vertices:
        for ll in f:
            if dim_ == 3:
                crack.write('%.15f %.15f %.15f\n' % (ll[0], ll[1], ll[2]))
            elif dim_ == 2:
                crack.write('%.15f %.15f 0.0\n' % (ll[0], ll[1]))
            nb_vertex += dim_
    crack.write("\n")

    point_tmp = 0;
    crack.write("CELLS %i %i\n" % (nb_facets, (dim_+ 1) * nb_facets))
    for i in range(nb_facets):
        crack.write('%i ' % dim_)
        for i in range(dim_):
            crack.write('%i ' % point_tmp)
            point_tmp += 1
        crack.write('\n')
    crack.write("\n")
    #
    crack.write("CELL_TYPES %i\n" % nb_facets)
    for i in range(nb_facets):
        if dim_ == 3:
            crack.write('5\n') #triangle
        elif dim_ == 2:
            crack.write('3\n') #edge #mettre 4 ?
    #
    crack.write("\n")
    crack.close()
    return

def local_gradient_matrix(num_cell, problem):
    res = lil_matrix((problem.d * problem.dim, problem.nb_dof_CR))
    bary_cell = problem.Graph.nodes[num_cell]['pos']
    vol = problem.Graph.nodes[num_cell]['measure']
    for (u,v) in nx.edges(problem.Graph, nbunch=num_cell): #getting facets of the cell (in edges)
        f = problem.Graph[u][v] #edge corresponding to the facet
        normal = f['normal']
        normal = normal * np.sign( np.dot(normal, f['barycentre'] - bary_cell) ) #getting the outer normal to the facet with respect to the cell
        dofs = f['dof_CR'] #getting the right number of the dof corresponding to the facet
        aux = np.zeros((problem.d * problem.dim, problem.d)) #local contribution to the gradient for facet
        if problem.d > 1: #vectorial problem
            for k in range(problem.d):
                for j in range(problem.d):
                    aux[k*problem.dim+j,k] = normal[j]
        else: #scalar problem
            for i in range(len(normal)):
                aux[i,0] = normal[i]
        aux = f['measure'] / vol * aux
        
        res[:, dofs[0] : dofs[problem.d-1] + 1] = aux #importation dans la matrice locale de la cellule
    return res

def removing_penalty(self, cracking_facets):
    dofmap_tens_DG_0 = self.W.dofmap()
    
    #creating jump matrix
    mat_jump_1 = dok_matrix((self.nb_dof_CR,self.nb_dof_DEM))
    mat_jump_2 = dok_matrix((self.nb_dof_CR,self.nb_dof_grad))

    for f in cracking_facets: #utiliser facet_num pour avoir les voisins ?
        assert len(self.facet_num.get(f)) == 2 
        c1,c2 = self.facet_num.get(f) #must be two otherwise external facet broke
        num_global_ddl = self.Graph[c1][c2]['dof_CR']
        coeff_pen = self.Graph[c1][c2]['pen_factor']
        pos_bary_facet = self.Graph[c1][c2]['barycentre'] #position barycentre of facet
        #filling-in the DG 0 part of the jump
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,self.d * c1 : (c1+1) * self.d] = np.sqrt(coeff_pen)*np.eye(self.d)
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,self.d * c2 : (c2+1) * self.d] = -np.sqrt(coeff_pen)*np.eye(self.d)

        for num_cell,sign in zip([c1,c2],[1., -1.]):
            #filling-in the DG 1 part of the jump...
            pos_bary_cell = self.Graph.nodes[num_cell]['pos']
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = np.sqrt(coeff_pen)*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                for i in range(self.dim):
                    mat_jump_2[dof_CR,tens_dof_position[(num % self.d)*self.d + i]] = sign*pen_diff[i]

    #Removing penalty
    self.mat_jump_1 -= mat_jump_1.tocsr()
    self.mat_jump_2 -= mat_jump_2.tocsr()
    return

def update_penalty_matrix(self):
    self.mat_jump_1.resize((self.nb_dof_CR,self.nb_dof_DEM))
    self.mat_jump_2.resize((self.nb_dof_CR,self.nb_dof_grad))
    self.mat_jump = self.mat_jump_1 + self.mat_jump_2 * self.mat_grad * self.DEM_to_CR
    self.mat_pen = self.mat_jump.T * self.mat_jump
    return

def energy_release_rates(self, vec_u_CR, cracked_facets, not_breakable_facets=set()):
    cracking_facets = set()
    stresses = self.mat_stress * self.mat_grad * vec_u_CR
    if self.d == 1:
        stress_per_cell = stresses.reshape((self.nb_dof_cells // self.d,self.dim))
    else:
        stress_per_cell = stresses.reshape((self.nb_dof_cells // self.d,self.dim,self.dim))

    #Computing Gh
    Gh = np.zeros(self.nb_dof_CR // self.d)
    for fp in cracked_facets:
        for f in self.facet_to_facet.get(fp) - not_breakable_facets:
            if len(self.facet_num.get(f)) == 2:
                c1,c2 = self.facet_num.get(f)
                c1p = self.facet_num.get(fp)[0]
                normal = self.Graph[c1p][self.nb_dof_cells // self.d + fp]['normal']
                vertices_facet = self.Graph[c1p][self.nb_dof_cells // self.d + fp]['vertices']
                dist_1 = min(np.linalg.norm(self.Graph.nodes[c1]['pos'] - vertices_facet[0]), np.linalg.norm(self.Graph.nodes[c1]['pos'] - vertices_facet[1]))
                #dist_2 = np.linalg.norm(self.Graph.nodes[c2]['pos'] - self.Graph[c1p][self.nb_dof_cells // self.d + fp]['barycentre'])
                dist_2 = min(np.linalg.norm(self.Graph.nodes[c2]['pos'] - vertices_facet[0]), np.linalg.norm(self.Graph.nodes[c2]['pos'] - vertices_facet[1]))
                stress_1 = np.dot(stress_per_cell[c1],normal)
                stress_2 = np.dot(stress_per_cell[c2],normal)
            if self.d == 1:
                G1 = stress_1 * stress_1
                G1 *= np.pi / self.mu * dist_1 #*2
                G2 = stress_2 * stress_2
                G2 *= np.pi / self.mu * dist_2 #*2
            else:
                G1 = np.dot(stress_1,stress_1)
                G1 *= np.pi / float(self.E) * dist_1
                G2 = np.dot(stress_2,stress_2)
                G2 *= np.pi / float(self.E) * dist_2

            Gh[f] = np.sqrt(G1*G2) #looks all right...
                
    return Gh

def energy_release_rates_bis(self, vec_u_CR, vec_u_DG):
    n = FacetNormal(self.mesh)
    S = FacetArea(self.mesh)
    u_CR = TrialFunction(self.CR)
    Dv_DG = TestFunction(self.W)

    #To get normal stress
    a = inner(dot(avg(Dv_DG),n('+')), u_CR('+')) / S('+') * dS
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    normal_stresses = A.T * self.mat_stress * self.mat_grad * vec_u_CR

    #To get facet jumps
    v_DG = TestFunction(self.DG_0)
    a = inner(jump(v_DG), u_CR('+')) / S('+') * dS
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    jumps = A.T * vec_u_DG

    #Assembling
    if self.d == 1:
        return np.pi * abs(normal_stresses * jumps)
    elif self.d == 2:
        normal_stresses = normal_stresses.reshape((self.initial_nb_dof_CR // self.d, self.dim))
        jumps = jumps.reshape((self.initial_nb_dof_CR // self.d, self.dim))
        return np.pi * sum(normal_stresses * jumps, axis=0)

def energy_release_rate_vertex(problem, broken_vertices, Gh_facets):
    res = np.zeros(problem.mesh.num_vertices())
    for v in broken_vertices:
        Gh_v = -1
        for f in problem.facets_vertex.get(v):
            Gh_v = max(Gh_v, Gh_facets[f])
        res[v] = Gh_v

    return res

def energy_release_rate_vertex_bis(problem, broken_vertices, broken_facets, vec_u_CR, vec_u_DG):
    res = np.zeros(problem.mesh.num_vertices())
    
    #To get facet jumps
    S = FacetArea(problem.mesh)
    u_CR = TrialFunction(problem.CR)
    v_DG = TestFunction(problem.DG_0)
    a = inner(jump(v_DG), u_CR('+')) / S('+') * dS
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))
    jumps = A.T * vec_u_DG

    #To get stresses in cells
    stresses = problem.mat_stress * problem.mat_grad * vec_u_CR
    stress = np.nan_to_num(stresses.reshape((problem.nb_dof_cells // problem.d, problem.dim)))
    
    for v in broken_vertices:
        Gh_v = -1
        for f in problem.facets_vertex.get(v):
            if f in broken_facets:
                n1 = problem.facet_num.get(f)[0]
                normal =  problem.Graph[n1][problem.nb_dof_cells // problem.d + f]['normal']
                for fp in problem.facets_vertex.get(v):
                    if len(problem.facet_num.get(fp)) == 2:
                        c1,c2 = problem.facet_num.get(fp)
                        Gh = np.pi*0.5*np.dot(stress[c1]+stress[c2],normal) * jumps[fp]
                        Gh_v = max(Gh_v, Gh)
        res[v] = Gh_v

    return res

def kinking_criterion(problem, v, vec_u_CR, not_breakable_facets):
    n = FacetNormal(problem.mesh)
    S = FacetArea(problem.mesh)
    u_CR = TrialFunction(problem.CR)
    Dv_DG = TestFunction(problem.W)

    #To get normal stress
    a = inner(dot(avg(Dv_DG),n('+')), u_CR('+')) / S('+') * dS
    A = assemble(a)
    row,col,val = as_backend_type(A).mat().getValuesCSR()
    A = csr_matrix((val, col, row))

    if problem.d == 1:
        normal_stresses = abs(A.T * problem.mat_stress * problem.mat_grad * vec_u_CR)
    elif problem.d == 2:
        normal_stresses = A.T * problem.mat_stress * problem.mat_grad * vec_u_CR
        normal_stresses = normal_stresses.reshape((problem.initial_nb_dof_CR // problem.d, problem.dim))
        #Calculer normals sur toutes les facettes !
        normal_stresses = np.sum(normal_stresses * normals, axis=0)
    breakable_facets = list(set(problem.facets_vertex.get(v)) - not_breakable_facets)
    print(breakable_facets)
    print(normal_stresses[breakable_facets])

    return breakable_facets[np.argmax(normal_stresses[breakable_facets])]
            
