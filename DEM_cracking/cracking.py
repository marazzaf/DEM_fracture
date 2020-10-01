#coding: utf-8

#Will contain all functions linked to cracking facets and adpating problem after cracking

def adapting_after_crack(cracking_facets, already_cracked_facets, d_, dim, face_num, nb_ddl_cells_, nb_ddl_ccG_, nb_ddl_CR, passage_CR, mat_grad, G_, mat_D, mat_not_D):
    nb_ddl_CR_new = nb_ddl_CR
    tetra_coord_bary = dict([])
    tetra_coord_num = dict([])
    impacted_facets = set([])

    #modifying connectivity graph
    set_impacted_cells = set() #filled-in in case of fragment detaching
    for f in cracking_facets:
        #print('num_face: %i' % f)
        n1,n2 = face_num.get(f) #n1 : plus, n2 : minus
        set_impacted_cells |= {n1,n2}
        #retrieving facets impacted by rutpure:
        impacted_facets.update(G_[n1][n2]['recon']) #impacted facets will have a new CR reconstruction
        ex_dofs = G_[n1][n2]['dof_CR']
        ex_num = G_[n1][n2]['num']
        ex_bary = G_[n1][n2]['barycentre']
        ex_normal = G_[n1][n2]['normal']
        ex_measure = G_[n1][n2]['measure']
        ex_vertices = G_[n1][n2]['vertices']
        ex_pen = G_[n1][n2]['pen_factor']
        
        #removing link between the two cell dofs
        G_.remove_edge(n1,n2)
        #adding the new facet dofs
        #dof_1 = list(np.arange(count_nb_ddl_ccG, count_nb_ddl_ccG+d_))
        G_.add_node(nb_ddl_cells_ // d_ + f, pos=ex_bary, sign=int(1), vertices=np.array([])) #linked with n1
        #count_nb_ddl_ccG += d_
        #dof_2 = list(np.arange(count_nb_ddl_ccG, count_nb_ddl_ccG+d_))
        G_.add_node(-nb_ddl_cells_ // d_ - f, pos=ex_bary, sign=int(-1), vertices=np.array([])) #linked with n2
        #count_nb_ddl_ccG += d_
        #adding the connectivity between cell dofs and new facet dofs
        G_.add_edge(n1, nb_ddl_cells_ // d_ + f, num=ex_num, dof_CR=ex_dofs, measure=ex_measure, barycentre=ex_bary, normal=ex_normal, breakable=False, vertices=ex_vertices, pen_factor = ex_pen)
        face_num[f] = [n1]
        new_dof_CR = list(np.arange(nb_ddl_CR_new, nb_ddl_CR_new+d_))
        G_.add_edge(-nb_ddl_cells_ // d_ - f, n2, num=nb_ddl_CR_new//d_, dof_CR=new_dof_CR, measure=ex_measure, barycentre=ex_bary, normal=ex_normal, breakable=False, vertices=ex_vertices, pen_factor = ex_pen)
        nb_ddl_CR_new += d_
        face_num[-f] = [n2]
        #end modifications in graph etc...

    #if fragment detaches modify mat_D and mat_not_D
    for c in set_impacted_cells:
        fragment = False
        count = 0
        for n in nx.neighbors(G_, c):
            if abs(n) >= nb_ddl_cells_: #boundary
                count += 1
        if count == dim+1:
            fragment = True

        if fragment: #modification of Schur complement
            #modify mat_D and mat_not_D
            #mat_D = mat_D.todok() #does not need to be modified
            mat_not_D = mat_not_D.todok()
            for i in G_.node[c]['dof']:
                mat_not_D[i,i] = 0. #Not used to solve system any longer
            mat_not_D = mat_not_D.tocsr()
            mat_not_D.eliminate_zeros()

        #How to take care of two cells detaching together ????
        

    #Modifying matrix passage_CR
    passage_CR.resize((nb_ddl_CR_new,nb_ddl_ccG_))
    passage_CR_new = passage_CR.tolil() #.todok()
    #fill rows for new dof
    #Putting ones in new facet dofs and putting 0 where there were barycentric reconstructions
    for f in cracking_facets:
        n1 = face_num.get(f)[0]
        n2 = face_num.get(-f)[0]
        num_global_ddl_1 = G_[n1][f + nb_ddl_cells_ // d_]['dof_CR']
        num_global_ddl_2 = G_[-f - nb_ddl_cells_ // d_][n2]['dof_CR']
            
        #deleting previous barycentric reconstruction
        passage_CR_new[num_global_ddl_1[0], :] = np.zeros(passage_CR_new.shape[1])
        if d_ >= 2: #deuxième ligne
            passage_CR_new[num_global_ddl_1[1], :] = np.zeros(passage_CR_new.shape[1])
        if d_ == 3: #troisième ligne
            passage_CR_new[num_global_ddl_1[2], :] = np.zeros(passage_CR_new.shape[1])
        #recomputing the CR reconstruction
        coord_bary_1,coord_num_1,connectivity_recon_1 = test_symmetric_bary_coord(f,face_num,dim,d_,G_,nb_ddl_cells_)
        coord_bary_2,coord_num_2,connectivity_recon_2 = test_symmetric_bary_coord(-f,face_num,dim,d_,G_,nb_ddl_cells_)

        #Filling-in the new reconstruction
        for i1,j1,i2,j2 in zip(coord_num_1,coord_bary_1,coord_num_2,coord_bary_2):
            passage_CR_new[num_global_ddl_1[0],i1[0]] += j1
            passage_CR_new[num_global_ddl_2[0],i2[0]] += j2 
            if d_ >= 2:
                passage_CR_new[num_global_ddl_1[1],i1[1]] += j1
                passage_CR_new[num_global_ddl_2[1],i2[1]] += j2
            if d_ == 3:
                passage_CR_new[num_global_ddl_1[2],i1[2]] += j1
                passage_CR_new[num_global_ddl_2[2],i2[2]] += j2

    impacted_facets.difference_update(already_cracked_facets) #removing already cracked facets from the set of facets that need a new CR reconstruction

    #The connectivity recon will be recomputed. Thus retrieving the num of the facet from the potentially impacted. Will be upadted afterwards.
    for (u,v,r) in G_.edges(data='recon'):
        if r != None:
            r.difference_update(impacted_facets)

    #computing the new CR reconstruction for impacted facets
    for f in impacted_facets:
        #if len(face_num.get(g)) > 1: #facet not on boundary otherwise reconstruction does not change
        coord_bary,coord_num,connectivity_recon = test_symmetric_bary_coord(f,face_num,dim,d_,G_,nb_ddl_cells_)
        tetra_coord_bary[f] = coord_bary
        tetra_coord_num[f] = coord_num

        #updating the connectivity_recon in the graph. Adding the new connectivity recon in graph.
        for k in connectivity_recon:
            if(len(face_num.get(k))) == 2:
                n1,n2 = face_num.get(k)
                G_[n1][n2]['recon'].add(f)

    #Putting-in the new barycentric coordinates for inner facet reconstruction that changed
    for f in impacted_facets:
        if len(face_num.get(f)) > 1: #facet not on boundary
            n1,n2 = face_num.get(f)
            num_global_ddl = G_[n1][n2]['dof_CR']
        else: #facet on boundary
            n = face_num.get(f)[0]
            num_global_ddl = G_[n][f + nb_ddl_cells_ // d_]['dof_CR']
        #erasing previous values
        passage_CR_new[num_global_ddl[0],:] = np.zeros(passage_CR_new.shape[1])
        if d_ >= 2:
            passage_CR_new[num_global_ddl[1],:] = np.zeros(passage_CR_new.shape[1])
        if d_ == 3:
            passage_CR_new[num_global_ddl[2],:] = np.zeros(passage_CR_new.shape[1])
            #Putting new values in
        for i,j in zip(tetra_coord_num.get(f),tetra_coord_bary.get(f)):
            passage_CR_new[num_global_ddl[0],i[0]] += j 
            if d_ >= 2:
                passage_CR_new[num_global_ddl[1],i[1]] += j
            if d_ == 3:
                passage_CR_new[num_global_ddl[2],i[2]] += j
            
    #Modifying the gradient reconstruction matrix
    #No renumbering because uses CR dofs and not ccG dofs !!!
    mat_grad.resize((mat_grad.shape[0],nb_ddl_CR_new))
    mat_grad_new = mat_grad.tolil() #only changes for cells having a facet that broke
    for f in cracking_facets:
        #first newly facet dof and associated cell
        c1 = face_num[f][0]
        mat_grad_new[c1 * d_ * dim : (c1+1) * d_ * dim, :] = local_gradient_matrix(c1, G_, dim, d_, nb_ddl_CR_new)
        c2 = face_num[-f][0]
        mat_grad_new[c2 * d_ * dim : (c2+1) * d_ * dim, :] = local_gradient_matrix(c2, G_, dim, d_, nb_ddl_CR_new)

    #Optimization
    passage_CR_back = passage_CR_new.tocsr()
    passage_CR_back.eliminate_zeros()


    return passage_CR_back.tocsr(), mat_grad_new.tocsr(), nb_ddl_CR_new, face_num, mat_D, mat_not_D

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

def local_gradient_matrix(num_cell, G_, dim, d_, nb_ddl_CR):
    res = sp.lil_matrix((d_ * dim, nb_ddl_CR))
    bary_cell = G_.node[num_cell]['pos']
    vol = G_.node[num_cell]['measure']
    for (u,v) in nx.edges(G_, nbunch=num_cell): #getting facets of the cell (in edges)
        f = G_[u][v] #edge corresponding to the facet
        normal = f['normal']
        normal = normal * np.sign( np.dot(normal, f['barycentre'] - bary_cell) ) #getting the outer normal to the facet with respect to the cell
        dofs = f['dof_CR'] #getting the right number of the dof corresponding to the facet
        aux = np.zeros((d_ * dim, d_)) #local contribution to the gradient for facet
        if d_ > 1: #vectorial problem
            for k in range(d_):
                for j in range(d_):
                    aux[k*dim+j,k] = normal[j]
        else: #scalar problem
            for i in range(len(normal)):
                aux[i,0] = normal[i]
        aux = f['measure'] / vol * aux
        
        res[:, dofs[0] : dofs[d_-1] + 1] = aux #importation dans la matrice locale de la cellule
    return res

def removing_penalty(mesh_, d_, dim_, nb_ddl_ccG_, mat_grad_, passage_ccG_CR_, G_, nb_ddl_CR_, cracking_facets, facet_num):
    if d_ >= 2:
        U_DG = VectorFunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = TensorFunctionSpace(mesh_, 'DG', 0)
    else:
        U_DG = FunctionSpace(mesh_, 'DG', 0)
        tens_DG_0 = VectorFunctionSpace(mesh_, 'DG', 0)
        
    nb_ddl_cells = U_DG.dofmap().global_dimension()
    dofmap_tens_DG_0 = tens_DG_0.dofmap()
    nb_ddl_grad = dofmap_tens_DG_0.global_dimension()

    #creating jump matrix
    mat_jump_1 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_ccG_))
    mat_jump_2 = sp.dok_matrix((nb_ddl_CR_,nb_ddl_grad))

    for f in cracking_facets: #utiliser facet_num pour avoir les voisins ?
        assert(len(facet_num.get(f)) == 2)
        c1,c2 = facet_num.get(f) #must be two otherwise external facet broke
        num_global_ddl = G_[c1][c2]['dof_CR']
        coeff_pen = G_[c1][c2]['pen_factor']
        pos_bary_facet = G_[c1][c2]['barycentre'] #position barycentre of facet
        #filling-in the DG 0 part of the jump
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c1 : (c1+1) * d_] = np.sqrt(coeff_pen)*np.eye(d_)
        mat_jump_1[num_global_ddl[0]:num_global_ddl[-1]+1,d_ * c2 : (c2+1) * d_] = -np.sqrt(coeff_pen)*np.eye(d_)

        for num_cell,sign in zip([c1,c2],[1., -1.]):
            #filling-in the DG 1 part of the jump...
            pos_bary_cell = G_.node[num_cell]['pos']
            diff = pos_bary_facet - pos_bary_cell
            pen_diff = np.sqrt(coeff_pen)*diff
            tens_dof_position = dofmap_tens_DG_0.cell_dofs(num_cell)
            for num,dof_CR in enumerate(num_global_ddl):
                for i in range(dim_):
                    mat_jump_2[dof_CR,tens_dof_position[(num % d_)*d_ + i]] = sign*pen_diff[i]

    return mat_jump_1.tocsr(), mat_jump_2.tocsr()

def cracking_criterion(problem):
    return 
