'''
Created on 2011-01-05

@author: kazem
'''
import networkx as nx
import random as rand
import csv
import math
import matplotlib.pylab as plt
import numpy as np
from Tree import binary_tree

try:
    set
except NameError:
    from sets import Set as set

class WCG:
    def __init__(self, n_):
        self.n = n_
        self.G = nx.Graph()
#        self.maps_ = {}
        #data structures needed for min hash
        self.k = 100
        self.primes = []
        self.a =[]
        self.b = []
    def __get_G(self):
        return self.G
    #this method plots H nicely
    def plot_graph(self, H):
        #find node labels depending if they are internal or external
        labels = {}
        nodelist_S = []
        nodelist_C = []
        cnt = 0
        for v in H.nodes():
            print v
            if v <= 99:
                labels[v] = 'S'
                nodelist_S.append(v)
            else:
                labels[v] = 'C'
                nodelist_C.append(v)
            cnt += 1
        pos = nx.spring_layout(H)
        print pos
        #pos = nx.random_layout(H)
        # define the color to select from the color map
        # as n numbers evenly spaced between color map limits
        node_color = map(int, np.linspace(0, 255, len(H.nodes())))
        # draw the nodes, specifying the color map and the list of color
        #nx.draw_networkx_nodes(H, pos,node_color=node_color, cmap=plt.cm.hsv)
        nx.draw_networkx_nodes(H, pos, nodelist_S, node_color='r', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(H, pos, nodelist_C, node_color='b', node_size=500, alpha=0.8)
        # add the labels inside the nodes
        #nx.draw_networkx_labels(H, pos)
        nx.draw_networkx_labels(H, pos, labels, font_size=16)
        # draw the edges, using alpha parameter to make them lighter
        nx.draw_networkx_edges(H, pos, alpha=0.4)
        # turn off axis elements
        plt.axis('off')
        plt.show()
    #construct the graph from csv input file
    def build_csv(self, in_file_, start_id, delim_):
        #reader = csv.reader(open("/home/kazem/Data/FB/facebook-links.txt"), delimiter="\t")
        reader = csv.reader(open(in_file_), delimiter=delim_)
        for line in reader:
                self.G.add_edge(int(line[0]) - start_id,int(line[1]) - start_id,weight=float(1))
        self.n = len(self.G.nodes())
        print "no of nodes: %d" % self.G.number_of_nodes()
        print "no of edges: %d" % self.G.number_of_edges()
	#comment the following line for generating a subgraph of G
	#return 0
    def remove_bridges(self, in_file_, start_id, delim_):
        reader = csv.reader(open(in_file_), delimiter=delim_)
        for line in reader:
                self.G.remove_edge(int(line[0]) - start_id,int(line[1]) - start_id)
	
	print "no of components after removing bridges: %d" % nx.number_connected_components(self.G)
	comps = nx.connected_components(self.G)
	for comp in comps:
        	print len(comp)

        bfs = self.BreadthFirstLevels(1,100)
        nbunch = [1]
        for n in bfs:
            #print(n)
            val_ = n.values()
            for set_ in val_:
                nbunch += list(set_)
        #print nbunch
        print "start creating the induced graph!"
        induced_g = nx.subgraph(self.G, nbunch)
        self.G.clear()
#        start_ = 0
#        for n_ in induced_g:
#            self.maps_[n_] = start_
#            start_ += 1
#        for n_1 in induced_g:
#            for n_2 in induced_g:
#                if n_1 in induced_g.neighbors(n_2):
#                    self.G.add_edge(maps_[n_1],maps_[n_2]) 
        self.n = nx.number_of_nodes(induced_g)
        self.G = induced_g
        print "no of node: %d and no of edges: %d in induce graph!" % (self.G.number_of_nodes(), self.G.number_of_edges())
    def findWhiskers(self, in_file_1, in_file_2, start_id, delim_):
    	#let's read all bridges
    	no_bri = 0
    	bridges = []
    	reader = csv.reader(open(in_file_1), delimiter=',')
        for line in reader:
            e_ = int(line[0]) - start_id,int(line[1]) - start_id
            bridges.append(e_)
            no_bri += 1
    	print "finished reading %d no of bridges" % no_bri 
    	#let's read all of nodes in the main component
        core = []
        reader = csv.reader(open(in_file_2), delimiter=delim_)
    	line_no = 0	#let's skip the first two lines 
    	core_size = 0
        for line in reader:
            if (line_no >= 2 and line_no < 36965):
                n_ = int(line[0]) - start_id
                core.append(n_)
                core_size += 1
            line_no += 1
    	print "size of core: %d" % core_size
    	if(75885 in core):
    		print "%d is in core" % 11429
    	if(58024 in core):
    		print "%d is in core" % 58024
    #	return 0 
    	#now let's find the main component of the original graph
        bfs = self.BreadthFirstLevels(1,100)
        nbunch = [1]
        for n in bfs:
            #print(n)
            val_ = n.values()
            for set_ in val_:
                nbunch += list(set_)
                #print nbunch
        print "start creating the induced graph!"
        induced_g = nx.subgraph(self.G, nbunch)
        self.G.clear()
        self.n = nx.number_of_nodes(induced_g)
        self.G = induced_g
    	
    	#now let's remove all bridges that just have one edge in core
    	print "identifying whiskers"
    	z = 0
    	for e_ in bridges:
    		if(z <= 3):
    			print "(%d,%d)" % (e_[0],e_[1])	
    		z += 1
    		if(((e_[0] in core) and (e_[1] not in core)) or ((e_[0] not in core) and (e_[1] in core))):
    			print "remove (%d,%d)" % (e_[0],e_[1])
    			self.G.remove_edge(e_[0], e_[1])
    
        print "no of components after removing bridges: %d" % nx.number_connected_components(self.G)
        comps = nx.connected_components(self.G)
        comp_num = 0
        for comp in comps:
#            if(len(comp) == 1):
#                comp_num += len(comp)
    		if(len(comp) > 1 and len(comp) < 1000):
    			print len(comp)
                comp_num += len(comp)
                print comp
        print "total comp size: %d" % comp_num 
    #Barbell graph with n nodes and comps components    
    def build_BarbellG(self, comps):
        c = 0 
        while (c < comps):
            for i in range(c*self.n/comps,(c + 1)*self.n/comps):
                for j in range(c*self.n/comps,(c + 1)*self.n/comps):
                    if i != j:
                        self.G.add_edge(i,j,weight=float(1))
            c += 1                
        #add a bridge
        c = 1
        while (c < comps):
            self.G.add_edge(c*self.n/comps - 1, c*self.n/comps, weight=float(1))
            c += 1
    #build three components that are connected by three different level of edges
    def build_tri(self):
        c = 0 
        comps = 3
        p_1 = 1.0/((math.log(self.n/comps)))
        #connect each component internally
        while (c < comps):
            for i in range(c*self.n/comps,(c + 1)*self.n/comps):
                for j in range(i + 1,(c + 1)*self.n/comps):
                    if rand.random() <= p_1:
                        self.G.add_edge(i,j,weight=float(1))
            c += 1
        #connect C1 to C2
        #p_2 = p_1/2.0#(self.n/comps)
        p_2 = 1.0/((self.n/comps)*math.sqrt(self.n/comps))
        for i in range(0*self.n/comps, (0 + 1)*self.n/comps):
            for j in range(1*self.n/comps, (1 + 1)*self.n/comps):
                if rand.random() <= p_2:
                    self.G.add_edge(i,j,weight=float(1))
        #connect C1 to C3
        #p_3 = p_1/4.0#(2*self.n/comps)
        p_3 = 4.0/((self.n/comps)*(self.n/comps))
        for i in range(0*self.n/comps, (0 + 1)*self.n/comps):
            for j in range(2*self.n/comps, (2 + 1)*self.n/comps):
                if rand.random() <= p_3:
                    self.G.add_edge(i,j,weight=float(1))
    #this method runs a simple rw starting from a random node        
    def calc_probs(self, u, steps):
        #we start from node u and compute the probabilities of moving to other components
        comps = 3
        c11 = c12 = c13 = 0
        rounds = steps
        while(rounds >= 1):
            #pick one of the edges out of node u
            nghs_ = self.G.neighbors(u)
            #let's pick a random neighbor from N(u)
            v = nghs_[rand.randint(0, len(nghs_) - 1)]
            print "move to %d" % v
            if v in range(0, self.n/comps):
                c11 += 1
            elif v in range(self.n/comps, 2*self.n/comps):
                c12 += 1
            elif v in range(2*self.n/comps, 3*self.n/comps):
                c13 += 1
            rounds -= 1
        print "(c11: %d, c12: %d, c13: %d)" % (c11, c12, c13)
        print "(p11: %f, p12: %f, p13: %f)" % (float(c11)/float(steps), float(c12)/float(steps), float(c13)/float(steps))                                                              
    #computing the mean of a list
    def mean(self,numberList):
        floatNums = [float(x) for x in numberList]
        if(len(numberList) > 0):
            return sum(floatNums) / len(numberList)
        else:
            return 0.0
    #build a hierarichal network based on Watts model
    #level: is the number of levels in Tree. For a tree with 8 leaves, the level is 4!
    #the average degree that we want to get for each node
    def build_hierarichal_watts(self, bran_ratio, level, alpha, avg_degree):
        c = alpha/(1 - math.exp(-alpha*level))
        #let's generate the all distances from all groups
        t = binary_tree()
        g_no = bran_ratio**(level - 1)    #number of groups
        print "no of groups: %d" % g_no
        print "tree's nodes ids are in the range of (%d,%d)" % (0, bran_ratio**level-1)
        for i in range(0, bran_ratio**level-1):
            t.add_node(i)
        #t.preorder()
        #let's enumerate all leaves first
        out_leaves = t.BFS_leaves(t.root)
        all_distances = []
        for v in out_leaves:
            #print "node : %d" % v.id
            dist = {}
            for d in range(1, level + 1):
                out = t.distance(v, d)
                #print "nodes that are at distance %d from %d:" % (d,v.id)
                dist[d] = out
                #for u in out: 
                #    print u
            all_distances.append(dist)
        #print all_distances
        #this is the list of all groups ids
        g_ids = []
        for u in out_leaves:
            g_ids.append(u.id)
        #print "g_ids: "
        #print g_ids
        #print "g_id[0] %d" % g_ids[0]
        deg_avg = 0.0  
        #let's generate links
        while deg_avg < avg_degree:
            #choose a random group for node i
            gi = rand.choice(g_ids)
            #print "random group gi: %d" % gi
            gi -= g_ids[0]
            #let's pick a random src for the link
            #print "picking i from range (%d,%d)" % (gi*self.n/g_no, (gi + 1)*self.n/g_no)
            i = rand.randrange(gi*self.n/g_no, (gi + 1)*self.n/g_no)
            #let's randomly generate a link with probability of c*exp(-alpha*x)
            x = (-1/alpha)*math.log(1-(alpha/c)*rand.uniform(0,1))
            #print "x: %f" % x
            d_ = math.ceil(x)   #the distance
            #let's find the corresponding dictionary for group gi (src node)
            dict = all_distances[gi]
            #print dict
            #let's pick a random group from dict[d_]
            #print "random dist: %d" % d_
            #print dict[d_]
            gj = rand.choice(dict[d_])
            gj -= g_ids[0]
            #let's pick a random dst for the link
            #print "picking j from range (%d,%d)" % (gj*self.n/g_no, (gj + 1)*self.n/g_no)            
            j = rand.randrange(gj*self.n/g_no, (gj + 1)*self.n/g_no)
            if i != j:
                #print "connecting i: %d to j: %d in G" % (i,j)
                self.G.add_edge(i,j,weight=float(1))
            deg_ = nx.degree(self.G).values()
            deg_avg = self.mean(deg_)    
    #create a clique with n nodes
    def build_Clique(self):
        self.G = nx.complete_graph(self.n)

    def build_watts_strogatz(self, k, p):
        self.G = nx.watts_strogatz_graph(self.n, k, p)
    def export_G_to_csv(self, flag):
	if(flag == 1):
        #now let's find the main component of the original graph
        	bfs = self.BreadthFirstLevels(0,100)
        	nbunch = [1]
        	for n in bfs:
            		#print(n)
            		val_ = n.values()
            		for set_ in val_:
                		nbunch += list(set_)
        	#print nbunch
        	print "start creating the induced graph!"
        	induced_g = nx.subgraph(self.G, nbunch)
        	self.G.clear()
        	self.n = nx.number_of_nodes(induced_g)
        	self.G = induced_g
        	#return

        print "starting writing no of nodes and edges"
        f = open('/Users/kazemjahanbakhsh/Downloads/epinion_core.csv', 'w')
        num_nodes = self.G.number_of_nodes()
        value = ''
        value += str(num_nodes)
        value += '\n' 
        f.write(value)        
        num_edges = self.G.number_of_edges()        
        value = ''
        value += str(num_edges)
        value += '\n'
        f.write(value)
        print "starting writing nodes ids and degrees"        
        nodes_ = self.G.nodes()
        for n_ in nodes_:
            value = ''
            value += str(n_)
	    value += ','
	    value += str(self.G.degree(n_))
            value += '\n' 
            f.write(value)
        print "starting storing adj lists"
        for n_ in nodes_:
            #print n_
            nghs_ = self.G.neighbors(n_)
            for ngh in nghs_:
                value = ''
                value = str(n_)
                value += ','
                value += str(ngh)
                value += ','
                value += str(1.0)
                value += '\n'
                f.write(value)
        f.close()
    def export_main_component(self, H):
        print "starting writing no of nodes and edges"
        f = open('/Users/kazemjahanbakhsh/Downloads/export_bar.csv', 'w')      
        print "starting storing adj lists"
        nodes_ = H.nodes()
        for n_ in nodes_:
            #print n_
            nghs_ = H.neighbors(n_)
            for ngh in nghs_:
                value = ''
                value = str(n_)
                value += '\t'
                value += str(ngh)
                value += '\n'
                f.write(value)
        f.close()
    def intersect(self, a, b):
        return list(set(a) & set(b))

    def union(self, a, b):
        return list(set(a) | set(b))

    def diff(self, a, b):
        #print a
        #print b
        #print set(a)
        #print set(b)
        return list(set(a) - set(b))
    
    def jacard_index(self, a, b):
        return float(len(list(set(a) & set(b))))/float(len(list(set(a) | set(b))))
    #this method spreads nodes' messages by running random walk in which each node u randomly picks one of its neighbors at each step and exchanges its message with it
    def spread_RW(self, debug_):
        flg_1 = flg_2 = flg_3 = 0   #for three components
        #the buffer from previous round
        Buff_prev = [[x] for x in self.G.nodes()]
        #the buffer for current round
        Buff_curr = [[x] for x in self.G.nodes()]
        cent_ = [[0 for x in self.G.nodes()] for x in range(0,len(self.G.nodes()))]
        r = 0;
        comp_= [0]*self.n
        done_ = 0
        while(done_ == 0):
            for v in self.G.nodes():
                if (comp_[v] == 0):
                    #node v has not finished yet
                    #pick one of the edges out of the starting_node with equal probs
                    nghs_ = self.G.neighbors(v)
                    #let's pick a random neighbor from N(v)
                    w = nghs_[rand.randint(0, len(nghs_) - 1)]
#                    print "node v: %d contacts w: %d" % (v,w)
#                    print "Current Buffer of v before exchange: "
#                    print Buff_curr[v]
                    #let's update their corresponding centrality counters by counting the number of new messages that we receive from one of our neigbors
                    cent_[w][v] += len(self.diff(Buff_prev[v], Buff_curr[w]))
                    cent_[v][w] += len(self.diff(Buff_prev[w], Buff_curr[v]))
                    #exchange messages between v and w
                    if(v == 1 and debug_):
                        print "round %d: no of updates from %d = %d (1-->w)" % (r, w, len(self.diff(Buff_prev[w], Buff_curr[v])))
                    if(w == 1 and debug_):
                        print "round %d: no of updates from %d = %d (w-->1)" % (r, v, len(self.diff(Buff_prev[v], Buff_curr[w])))

                    #we should see what are in v's current buffer and contacted node previous buffer
                    Buff_curr[v] = self.union(Buff_curr[v], Buff_prev[w])
                    Buff_curr[w] = self.union(Buff_prev[v], Buff_curr[w])
#                    print "Current Buffer of v after exchange: "
#                    print Buff_curr[v]                    
                    #let's check if v is done    
                    if(len(Buff_curr[v]) == self.n):
                        comp_[v] = 1
#                        print "%d is done!" % v
            #now one round is finished
            r += 1
#            print "round #: %d" % r
            Buff_prev = Buff_curr
            #let's check if we are done
            done_ = 1 
            for v in self.G.nodes():
                if (comp_[v] == 0):
                    done_ = 0
                    break
            #for node y=0, let's see when we are done with different components
            y = 0
            comps = 3
            done_1 = 1
            if(flg_1 == 0):
                for z in range(0, self.n/comps):
                    if(z not in Buff_curr[y]):
                        done_1 = 0
                        break
            if(done_1 == 1 and flg_1 == 0):
                print "c11 done at round %d" % r                
                flg_1 = 1
            done_2 = 1
            if(flg_2 == 0):
                for z in range(self.n/comps, 2*self.n/comps):
                    if(z not in Buff_curr[y]):
                        done_2 = 0
                        break
            if(done_2 == 1 and flg_2 == 0):
                print "c12 done at round %d" % r
                flg_2 = 1
            done_3 = 1
            if(flg_3 == 0):
                for z in range(2*self.n/comps, 3*self.n/comps):
                    if(z not in Buff_curr[y]):
                        done_3 = 0
                        break
            if(done_3 == 1 and flg_3 == 0):
                print "c13 done at round %d" % r
                flg_3 = 1
#        max_ = 0
#        inx_ = 0
#        print "node : %d" % inx_
#        for i in range(0,self.n):
#            print "the centrality of %d is %d" % (i,cent_[0][i])
#            if (cent_[0][i] > max_):
#                max_ = cent_[0][i]
#                inx_ = i
#        max_ = 0
#        print "node : %d" % inx_
#        for i in range(0,self.n):
#            print "the centrality of %d is %d" % (i,cent_[inx_][i])
#            if (cent_[inx_][i] > max_):
#                max_ = cent_[inx_][i]
#                inx_ = i
#        print "node : %d" % inx_
#        for i in range(0,self.n):
#            print "the centrality of %d is %d" % (i,cent_[inx_][i])
#        for i in range(0,self.n):
#            print Buff_curr[i]
        summed = [0]*self.n
        for i in range(0,self.n):
            #print "node %d" % i
            #print cent_[i]
            summed = [sum(pair) for pair in zip(cent_[i], summed)] 
        summed = [i/(float(self.n)**2) for i in summed]
        print "average cent based on RW:"
        print summed
        return r
    def spread_Censor(self, debug_):
        #the buffers structures for all nodes
        Buff_prev = [[x] for x in range(0,len(self.G.nodes()))]
        Buff_curr = [[x] for x in range(0,len(self.G.nodes()))]
        #a list that maintains the bottleneck nodes
        Bottlenecks_ = [[x for x in self.G.neighbors(x)] for x in range(0,len(self.G.nodes()))]
        #randomly shuffle the Bottleneck list
        for v in self.G.nodes():
            rand.shuffle(Bottlenecks_[v])
        #a list of the next nodes from Bottleneck lists to be contacted by each node
        nxt_cont = [0 for v in self.G.nodes()]
        for v in self.G.nodes():
            #print v
            #print Bottlenecks_[v]
            nxt_cont[v] = Bottlenecks_[v][0]
#        print Bottlenecks_
#        print nxt_cont
        #let's keep the neighbors list for all nodes
        nghs_ = [self.G.neighbors(v) for v in self.G.nodes()]
        #let's keep the no of neighbors for each node
        ngh_len = [len(self.G.neighbors(v)) for v in self.G.nodes()]
        #a list of contacted events for each round
        contact_list = [[-1] for x in range(0,len(self.G.nodes()))]
        r = 0;
        # a list that shows which nodes are done!
        completed_= [0]*self.n
        done_ = 0
        while(done_ == 0):
            for v in self.G.nodes():
                if (completed_[v] == 0):
                    #node v has not finished yet
                    if r%2 == 0:
                        #pick one of the edges out of the starting_node with equal probs
                        #nghs_ = self.G.neighbors(v)
                        #let's pick a random neighbor from N(v)
                        w = nghs_[v][rand.randint(0, ngh_len[v] - 1)]
                        if debug_:
                            print "round %d & v: %d contacts w: %d" % (r, v, w)
                        #add the w into v's contact list
                        contact_list[v] = w
                    else:
                        #pick the next node from Bt(v)
                        if debug_:
                            print "Bottleneck list of %d" % v
                        if debug_:
                            print Bottlenecks_[v]
                        w = nxt_cont[v]
                        if debug_:
                            print "round %d & v: %d contacts w: %d" % (r, v, w)
                        #add the w into v's contact list
                        contact_list[v] = w
                        #compute the next node that v contacts from Bottleneck list
                        #first find the current contacted node index in Bottelneck list
                        inx_ = Bottlenecks_[v].index(w)
                        #now find the next node from the list to contact the next odd round #
                        nxt_cont[v] = Bottlenecks_[v][(inx_+1)%len(Bottlenecks_[v])]
                else:
                    #v is done, so it doesn't contact anybody
                    contact_list[v] = -1
            for v in self.G.nodes():
                if (completed_[v] == 0):
                    #node v has not finished yet
                    for vi in self.G.nodes():
                        if (contact_list[v] == vi or contact_list[vi] == v): 
                            if debug_:
                                print "node v: %d contacts vi: %d" % (v,vi)
                                print "Current Buffer of v before exchange: "
                                print Buff_curr[v]
                            #we have to update Mt(v) by adding vi's messages to v's
                            #we should first find the list of new messages that we receive from vi for the first time
                            new_msgs = self.diff(Buff_prev[vi], Buff_curr[v])
                            for u in new_msgs:
                                if (u in self.G.neighbors(v) and u != contact_list[v]):
                                    #we need to check the above condition to make sure this u is not a node that v directly contacts
                                    if ((vi == contact_list[v] and vi != u) or contact_list[vi] == v):
                                        #if u is the next node to be contacted from Bottle, we need to compute a new node to be contacted next
                                        if u == nxt_cont[v]:
                                            #first find the current contacted node index in Bottelneck list
                                            inx_ = Bottlenecks_[v].index(u)
                                            #now find the next node from the list to contact the next odd round #
                                            nxt_cont[v] = Bottlenecks_[v][(inx_+1)%len(Bottlenecks_[v])]
                                        #now safely remove u from v's bottleneck list
                                        if debug_:
                                            print "remove u: %d from v: %d's bottleneck list" % (u,v)
                                            print "bottleneck list before removal:"
                                            print Bottlenecks_[v]
                                        Bottlenecks_[v].remove(u)
                                        if debug_:
                                            print "bottleneck list after removal:"
                                            print Bottlenecks_[v]
                            #after updating Bottleneck list now let's update Mt(v)
                            Buff_curr[v] = self.union(Buff_curr[v], Buff_prev[vi])
                            if debug_:
                                print "Current Buffer of v after exchange: "
                                print Buff_curr[v]                    
                    #let's check if v is done    
                    if(len(Buff_curr[v]) == self.n):
                        completed_[v] = 1
            #now one round is finished
            r += 1
            if debug_:
                print "round #: %d" % r
            Buff_prev = Buff_curr
            #let's check if we are done
            done_ = 1 
            for v in self.G.nodes():
                if (completed_[v] == 0):
                    done_ = 0
                    break   
        #for i in range(0,self.n):
            #print "(%d,%s)" % (i, Bottlenecks_[i])
            #if(i == self.n/2 or i == self.n/2 - 1):
                    #print "(%d,%d)" % (i,len(Bottlenecks_[i]))
        
        return r
        dic = {}
        dic[0] = Bottlenecks_[0]
        dic[1] = Bottlenecks_[1]
        #return dic
    def spread_wties(self, debug_):
        #the buffers structures for all nodes
        Buff_prev = [[] for x in range(0,64000)]
        Buff_curr = [[] for x in range(0,64000)]
        #node 1 just has a single message that wants to broadcast it
        Buff_prev[1] = [1]
        Buff_curr[1] = [1]
        len(Buff_curr)
        #a list that maintains the bottleneck nodes
        #Bottlenecks_ = [[x for x in self.G.neighbors(x)] for x in range(0,len(self.G.nodes()))]
        Bottlenecks_ = []
        for x in range(0, 64000):
            if (x in self.G):
                Bottlenecks_.append(self.G.neighbors(x))
            else:
                Bottlenecks_.append([-1])
                            
        #randomly shuffle the Bottleneck list
        print len(Bottlenecks_)
        for v in self.G.nodes():
            #print v
            rand.shuffle(Bottlenecks_[v])
        #a list of the next nodes from Bottleneck lists to be contacted by each node
        nxt_cont = [0 for v in range(0, 64000)]
        for v in self.G.nodes():
            #print v
            #print Bottlenecks_[v]
            nxt_cont[v] = Bottlenecks_[v][0]
#        print Bottlenecks_
#        print nxt_cont
        #let's keep the neighbors list for all nodes
        #nghs_ = [self.G.neighbors(v) for v in self.G.nodes()]
        nghs_ = []
        for v in range(0, 64000):
            if v in self.G:
                nghs_.append(self.G.neighbors(v))
            else:
                nghs_.append([-1])
        #let's keep the no of neighbors for each node
        #ngh_len = [len(self.G.neighbors(v)) for v in self.G.nodes()]
        ngh_len = []
        for v in range(0, 64000):
            if v in self.G:
                ngh_len.append(len(self.G.neighbors(v)))
            else:
                ngh_len.append([0])
        #a list of contacted events for each round
        #contact_list = [[-1] for x in range(0,len(self.G.nodes()))]
        contact_list = []
        for v in range(0, 64000):
            contact_list.append([-1])

        r = 0;
        # a list that shows which nodes are done!
        completed_= [0]*64000#self.n
        done_ = 0
        while(done_ == 0):
            for v in self.G.nodes():
                if (completed_[v] == 0):
                    #node v has not finished yet
                    if r%2 == 0 or True:
                        #pick one of the edges out of the starting_node with equal probs
                        #nghs_ = self.G.neighbors(v)
                        #let's pick a random neighbor from N(v)
                        w = nghs_[v][rand.randint(0, ngh_len[v] - 1)]
                        if debug_:
                            print "round %d & v: %d contacts w: %d" % (r, v, w)
                        #add the w into v's contact list
                        contact_list[v] = w
#                    else:
#                        #pick the next node from Bt(v)
#                        if debug_:
#                            print "Bottleneck list of %d" % v
#                        if debug_:
#                            print Bottlenecks_[v]
#                        w = nxt_cont[v]
#                        if debug_:
#                            print "round %d & v: %d contacts w: %d" % (r, v, w)
#                        #add the w into v's contact list
#                        contact_list[v] = w
#                        #compute the next node that v contacts from Bottleneck list
#                        #first find the current contacted node index in Bottelneck list
#                        inx_ = Bottlenecks_[v].index(w)
#                        #now find the next node from the list to contact the next odd round #
#                        nxt_cont[v] = Bottlenecks_[v][(inx_+1)%len(Bottlenecks_[v])]
                else:
                    #v is done, so it doesn't contact anybody
                    contact_list[v] = -1
            for v in self.G.nodes():
                print v
                if (completed_[v] == 0):
                    #node v has not finished yet
                    for vi in self.G.nodes():
                        if (contact_list[v] == vi or contact_list[vi] == v): 
                            if debug_ and len(Buff_prev[v]) != 0:
                                print "node v: %d contacts vi: %d" % (v,vi)
                                print "Current Buffer of v before exchange: "
                                print Buff_curr[v]
                            #we have to update Mt(v) by adding vi's messages to v's
                            #we should first find the list of new messages that we receive from vi for the first time
                            new_msgs = self.diff(Buff_prev[vi], Buff_curr[v])
                            for u in new_msgs:
                                if (u in self.G.neighbors(v) and u != contact_list[v]):
                                    #we need to check the above condition to make sure this u is not a node that v directly contacts
                                    if ((vi == contact_list[v] and vi != u) or contact_list[vi] == v):
                                        #if u is the next node to be contacted from Bottle, we need to compute a new node to be contacted next
                                        if u == nxt_cont[v]:
                                            #first find the current contacted node index in Bottelneck list
                                            inx_ = Bottlenecks_[v].index(u)
                                            #now find the next node from the list to contact the next odd round #
                                            nxt_cont[v] = Bottlenecks_[v][(inx_+1)%len(Bottlenecks_[v])]

                            #after updating Bottleneck list now let's update Mt(v)
                            if len(Buff_prev[vi]) != 0:
                                Buff_curr[v] = self.union(Buff_curr[v], Buff_prev[vi])
                            if debug_ and len(Buff_curr[v]) != 0:
                                print "Current Buffer of v after exchange: "
                                print Buff_curr[v]                    
                    #let's check if v is done    
                    if(len(Buff_curr[v]) == 1):
                        completed_[v] = 1
                        print "one completion"
            #now one round is finished
            r += 1
            if debug_:
                print "round #: %d" % r
            Buff_prev = Buff_curr
            #let's check if we are done
            done_ = 1 
            for v in self.G.nodes():
                if (completed_[v] == 0):
                    done_ = 0
                    break   
        #for i in range(0,self.n):
            #print "(%d,%s)" % (i, Bottlenecks_[i])
            #if(i == self.n/2 or i == self.n/2 - 1):
                    #print "(%d,%d)" % (i,len(Bottlenecks_[i]))
        
        return r            
    #this method runs a simple rw starting from a random node        
    def rw(self, step):
        #we start from a randomly chosen node
        nodes_ = self.G.nodes()
        v = nodes_[rand.randint(0, len(nodes_)-1)]
        print "starting node: "
        print v
        while(step >= 1):
            #pick one of the edges out of node v
            nghs_ = self.G.neighbors(v)
            #let's pick a random neighbor from N(v)
            v = nghs_[rand.randint(0, len(nghs_) - 1)]
            print "move to %d" % v
            step -= 1
    """    
    Breadth First Search.
    D. Eppstein, May 2007.
    """
    def BreadthFirstLevels(self, root, max_level):
        #print "hi from bfs"
        """
        Generate a sequence of bipartite directed graphs, each consisting
        of the edges from level i to level i+1 of G. Edges that connect
        vertices within the same level are not included in the output.
        The vertices in each level can be listed by iterating over each
        output graph.
        """
        level = 0
        visited = set()
        currentLevel = [root]
        while currentLevel and level <= max_level:
            for v in currentLevel:
                visited.add(v)
            nextLevel = set()
            levelGraph = dict([(v,set()) for v in currentLevel])
            for v in currentLevel:
                for w in self.G.neighbors(v):
                    if w not in visited:
                        levelGraph[v].add(w)
                        nextLevel.add(w)
            yield levelGraph
            currentLevel = nextLevel
            level += 1
            #print level
        print level
    
    # prime numbers are only divisible by unity and themselves
    # (1 is not considered a prime number by convention)
 
    def isprime(self, n):
        '''check if integer n is a prime'''
        # make sure n is a positive integer
        n = abs(int(n))
        # 0 and 1 are not primes
        if n < 2:
            return False
        # 2 is the only even prime number
        if n == 2:
            return True
        # all other even numbers are not primes
        if not n & 1:
            return False
        # range starts with 3 and only needs to go up the squareroot of n
        # for all odd numbers
        for x in range(3, int(n**0.5)+1, 2):
            if n % x == 0:
                return False
        return True
    
    """
    setup primes and a and b arrays
    """
    def init_minhash(self):
        self.k = 100
        prime_count = self.k
        #let's generate count primes
        cur_prime = 15485863
        while (prime_count > 0):
            if self.isprime(cur_prime) == True:
                self.primes.append(cur_prime)
                prime_count = prime_count - 1
            cur_prime = cur_prime + 1
        #print self.primes
        #print len(self.primes)
        #first generate the triple (a,b,P)

        for i in range(self.k):
            self.a.append(rand.randrange(1, self.primes[i] - 1))
            self.b.append(rand.randrange(1, self.primes[i] - 1))
    """
    compute the signature of a set
    """
    def compute_sig(self, alist):
        #let's compute the signature of a set
        list_sig = []
        for j in range(self.k):
            min_hash = self.primes[j]
            for i in alist:
                #compute the linear permutation
                cur_perm = (self.a[j]*i + self.b[j]) % self.primes[j]
                #print "%d %d %d" % (a[j], b[j], primes[j])
                #print cur_perm
                #print "----------"
                if cur_perm < min_hash:
                    min_hash = cur_perm
            #print min_hash        
            list_sig.append(min_hash)
        return list_sig
    
    def compute_minwise_hash(self, l1_sig, l2_sig):
        #lets compute minwise hash
        minwise = 0
        for i in range(self.k):
            if (l1_sig[i] == l2_sig[i]):
                minwise = minwise + 1
        #print minwise
        #print float(minwise)/float(self.k)
        return float(minwise)/float(self.k)
    """    
    test Minhash wise computation for two input sets.
    K. Jahanbakhsh, Jan 2012.
    """
    def test_minhash(self, list_1, list_2):
        #initialize the required data structures
        self.init_minhash()
        
        #compute minhash of the first set
        #print a
        #print b
        
        l1_sig = self.compute_sig(list_1)            
        l2_sig = self.compute_sig(list_2)

        #print l1_sig
        #print l2_sig
        
        sim = self.compute_minwise_hash(l1_sig, l2_sig)
        print sim
    #this method finds how long it takes for a node to collect all its neighbors messages by rw
    def spread_Ngh(self, src, debug_):
        #the buffer from previous round
        src_nghs = self.G.neighbors(src)
        src_ngh_len = len(src_nghs)
        indirect_ = 0
        direct_ = 0
        #weights of nodes that we receive their message from them directly
        dir_weights = []
        #weights of nodes that we receive their message indirectly
        ind_weights = []
        weights = []
        Buff_prev = [[] for x in range(0,64000)]
        Buff_curr = [[] for x in range(0,64000)]
        #only neighbors of v has a message
        for u_ in src_nghs:
            Buff_prev[u_] = [u_]
            Buff_curr[u_] = [u_]
        r = 0;
        done_ = 0
        print "round #: %d" % r
        while(done_ == 0):
            for v in self.G.nodes():
                #pick one of the edges out of the starting_node with equal probs
                nghs_ = self.G.neighbors(v)
                #let's pick a random neighbor from N(v)
                w = nghs_[rand.randint(0, len(nghs_) - 1)]
                if((v == src and w in src_nghs) or (w == src and v in src_nghs)):
                    print "node v: %d contacts w: %d" % (v,w)
#               print "Current Buffer of v before exchange: "
#               print Buff_curr[v]
                #exchange messages between v and w
                #check direct receives and indirect receives
                if v == src:
                    for u_ in src_nghs:
                        if u_ in Buff_prev[w] and u_ not in Buff_prev[v]:
                            #first time receive condition
                            weights.append(self.jacard_index(self.G.neighbors(src), self.G.neighbors(u_)))
                            if u_ == w:
                                #src receives m(u_) when src contacts u_ directly
                                print "src-->w: %d receives m(%d) directly" % (v,u_)
                                direct_ += 1
                                dir_weights.append(self.jacard_index(self.G.neighbors(src), self.G.neighbors(u_)))
                            else:
                                print "src-->w: %d receives m(%d) indirectly from %d" % (v,u_,w)
                                indirect_ += 1
                                ind_weights.append(self.jacard_index(self.G.neighbors(src), self.G.neighbors(u_)))
                if w == src:
                    for u_ in src_nghs:
                        if u_ in Buff_prev[v] and u_ not in Buff_prev[w]:
                            weights.append(self.jacard_index(self.G.neighbors(src), self.G.neighbors(u_)))
                            #first time receive condition
                            if u_ == v:
                                #src receives m(u_) when u_ contacts src directly
                                print "v-->src: %d receives m(%d) directly" % (w,u_)
                                direct_ += 1
                                dir_weights.append(self.jacard_index(self.G.neighbors(src), self.G.neighbors(u_)))
                            else:
                                print "v-->src: %d receives m(%d) indirectly from %d" % (w,u_,v)
                                indirect_ += 1
                                ind_weights.append(self.jacard_index(self.G.neighbors(src), self.G.neighbors(u_)))
                #we should see what are in v's current buffer and contacted node previous buffer
                if (len(Buff_prev[w]) != 0):
                    Buff_curr[v] = self.union(Buff_curr[v], Buff_prev[w])
                if (len(Buff_prev[v]) != 0):
                    Buff_curr[w] = self.union(Buff_prev[v], Buff_curr[w])
                   
            Buff_prev = Buff_curr
            print "Current Buffer of src after exchange: "
            print Buff_curr[src]
            print len(Buff_curr[src])
            #now one round is finished
            r += 1
            print "round #: %d" % r
            #let's check if we are done
            if (len(Buff_curr[src]) == src_ngh_len):
                done_ = 1
                print "direct: %d out of %d" % (direct_, src_ngh_len)
                print "indirect: %d out of %d" % (indirect_, src_ngh_len)
                print "weights:"
                print weights
                print "direct weights:"
                print dir_weights
                print "indirect weights:"
                print ind_weights
        return r
        
            