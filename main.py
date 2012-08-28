'''
Created on 2011-01-05

@author: kazem
'''
from WCG import WCG
from Tree import binary_tree
import networkx as nx
import pkg_resources
pkg_resources.require("matplotlib")
import matplotlib.pylab as plt
import math
import numpy as np
import string
import csv
import sys
#exporting the FB graph
##getting the core or the 2-edge connected component
def pair_compare(x,y):
	if x[1] > y[1]:
		return 1
	elif x[1] == y[1]:
	    return 0
	else:
	    return -1

def pair_inv_compare(x,y):
	if x[1] < y[1]:
		return 1
	elif x[1] == y[1]:
	    return 0
	else:
	    return -1	   
if(0):
	g = WCG(2048)
	g.build_csv("facebook-links.txt", 0, "\t")
	print "no of components before removing bridges: %d" % nx.number_connected_components(g.G)
	comps = nx.connected_components(g.G)
	#for comp in comps:
        #	print len(comp)
	#g.remove_bridges("/home/kazem/weakc/out",0,",")
	g.export_G_to_csv(1)
	#diam_ = nx.algorithms.distance_measures.diameter(g.G)
	#print "diameter of G: %d" %diam_
	
if(0):
	g = WCG(100)
	edge_thr = 1800
	#let's read the contact duration between pairs
	reader = csv.reader(open("/Users/kazemjahanbakhsh/Downloads/graph_cd.txt"), delimiter=',')
	for line in reader:
		if float(line[2]) > edge_thr:
			g.G.add_edge(int(line[0]),int(line[1]),weight=float(line[2]))
	#plot G
	#g.plot_graph(g.G)
	c = list(nx.k_clique_communities(g.G, 4))
	print c
if(0):
    g = WCG(100)
    g.build_csv("/home/kazem/data/FB/facebook-links.txt", 0, "\t")
	#g.findWhiskers("/home/kazem/weakc/out","/home/kazem/weakc/export.csv",0,",")
    #H = g.G.subgraph([60512, 60513, 60514, 60515, 60516, 60517, 60518, 60519, 60520, 60508, 60509, 60510, 60511])
    #H = g.G.subgraph([63008, 63687, 54988, 54989, 54990, 54991, 55065, 55066, 63356, 63357, 63007])
    #H = g.G.subgraph([54380, 54381, 54382, 54383, 54384, 54385, 54386, 54387, 54388, 35581])
    #H = g.G.subgraph([62496, 60099, 60100, 45811, 45812, 45813, 50652, 61054, 61055])
    #H = g.G.subgraph([48870, 48871, 48872, 49773, 49774, 50609, 50610, 50611, 50612])
    H = g.G.subgraph([45958, 45959, 45960, 45961, 45962, 45963, 49551, 49082, 49083, 49084, 49085, 49086, 59970, 59971, 51115, 51116, 62238, 62239, 61477, 55530, 61143, 59418, 59419, 59583, 59055, 59052, 59053, 59054, 41127, 47426, 47427, 47428, 47429, 57631, 49120, 49121, 49122, 54528, 49119, 63321, 59490, 52851, 52852, 63483, 52888, 52889, 52890, 59523, 59522, 54816, 60288, 62493, 62437, 60287, 59792, 59793, 59794, 59790, 59791, 37891, 37892, 15773, 36262, 59569, 60724, 60614, 28727, 61408, 55076, 55075, 29972, 48870, 48871, 48872, 49773, 49774, 50609, 50610, 50611, 50612, 62496, 60099, 60100, 45811, 45812, 45813, 50652, 61054, 61055, 54380, 54381, 54382, 54383, 54384, 54385, 54386, 54387, 54388, 35581, 63008, 63687, 54988, 54989, 54990, 54991, 55065, 55066, 63356, 63357, 63007, 60512, 60513, 60514, 60515, 60516, 60517, 60518, 60519, 60520, 60508, 60509, 60510, 60511, 57953, 57801, 57802, 57946, 53501, 53502, 53503])

if(0):
    g = WCG(2048)
    g.build_csv("/Users/kazemjahanbakhsh/Downloads/facebook-links.txt", 0, "\t")
    print "no of components before removing bridges: %d" % nx.number_connected_components(g.G)
    comps = nx.connected_components(g.G)
    for comp in comps:
            print len(comp)
    #g.export_G_to_csv(1)
    H = nx.connected_component_subgraphs(g.G)[0]
    g.export_main_component(H)
#    H = nx.connected_component_subgraphs(g.G)[0]
#    print nx.number_of_nodes(H)
#    l1 = g.G.neighbors(1)
#    l1.sort()
#    print l1
#    l2 = g.G.neighbors(2)
#    l2.sort()
#    print l2
#    g.minhash(g.G.neighbors(1), g.G.neighbors(2))

if(0):
	g = WCG(2048)
	g.build_csv("/Users/kazemjahanbakhsh/Downloads/export_bar.csv", 0, "\t")
	print nx.number_of_nodes(g.G)
	l1 = g.G.neighbors(1)
	l1.sort()
	print l1
	l2 = g.G.neighbors(2)
	l2.sort()
	print l2
	g.test_minhash(g.G.neighbors(1), g.G.neighbors(2))
	#let's go through all nodes and compute their adjaceny list signature
	g.init_minhash()
	signatures = []
	print "go through all nodes"
	nodes_ = g.G.nodes()
	map_index = {}
	cur_inx = 0
	for n_ in nodes_:
		#print n_
		adj_ = g.G.neighbors(n_)
		signatures.append(g.compute_sig(adj_))
		map_index[n_] = cur_inx
		cur_inx += 1
#let's extract the edge strenghts
	edges_strength = []
	for e_ in g.G.edges():
		#print e_
		min_hash = g.compute_minwise_hash(signatures[map_index[e_[0]]], signatures[map_index[e_[1]]])
		edges_strength.append(min_hash)
	#let's plot the histogram of ties strength
	plt.hist(edges_strength, 10)
	plt.show()
	#sys.exit(0)
#	print signatures[map_index[1]]
#	print signatures[map_index[2]]
#	print g.compute_minwise_hash(signatures[map_index[1]], signatures[map_index[2]])
	#for all edges let's compute the minwise hash
#	adj_ = g.G.neighbors(1)
#	for n_ in adj_:
#		print n_
#		print adj_
#		print g.G.neighbors(n_)
#		print g.compute_minwise_hash(signatures[map_index[1]], signatures[map_index[n_]])
	
#	file_ = open('/Users/kazemjahanbakhsh/Downloads/bridges.csv', 'w')
#	print "starting storing bridges"
#	print "length of sigs %d" % len(signatures)
#	for e_ in g.G.edges():
#		#print e_
#		value = ''
#		value = str(e_[0])
#		value += '\t'
#		value += str(e_[1])
#		value += '\t'
#		value += str(g.compute_minwise_hash(signatures[map_index[e_[0]]], signatures[map_index[e_[1]]]))
#		value += '\n'
#		file_.write(value)
#	
#	file_.close()
	print "start writing no of nodes and edges"
	f = open('/Users/kazemjahanbakhsh/Downloads/weighted_adj.csv', 'w')
	num_nodes = g.G.number_of_nodes()
	value = ''
	value += str(num_nodes)
	value += '\n' 
	f.write(value)		
	num_edges = g.G.number_of_edges()		
	value = ''
	value += str(num_edges)
	value += '\n'
	f.write(value)
	print "start writing nodes ids and degrees"		
	nodes_ = g.G.nodes()
	for n_ in nodes_:
		value = ''
		value += str(n_)
		value += ','
		value += str(g.G.degree(n_))
		value += '\n' 
		f.write(value)
	print "start storing adj lists"
	for n_ in nodes_:
		#print n_
		nghs_ = g.G.neighbors(n_)
		pair_list = []
		for ngh in nghs_:
			#let's create the list of pairs (node_id, weight)
			pair_ = (ngh, g.compute_minwise_hash(signatures[map_index[n_]], signatures[map_index[ngh]]))
			pair_list.append(pair_)
		#sort the list of weights
		pair_list.sort(pair_compare)
		#now store the sorted list
		for pair_ in pair_list:
			value = ''
			value = str(n_)
			value += ','
			value += str(pair_[0])
			value += ','
			value += str(pair_[1])
			value += '\n'
			f.write(value)
	f.close()
#	print "computing the histogram"
#	histogram_ = [0]*10
#	zeros_ = 0
#	for e_ in g.G.edges():
#		#print e_
#		min_hash = g.compute_minwise_hash(signatures[map_index[e_[0]]], signatures[map_index[e_[1]]])
#		histogram_[int(math.floor(min_hash*10))] += 1
#		if min_hash == 0.0:
#			zeros_ += 1
#	print "done!"
#	print histogram_
#	print zeros_
if(0):
	g = WCG(2048)
	g.build_csv("/Users/kazemjahanbakhsh/Downloads/facebook-links.txt", 0, "\t")
	u = 17811
	nghs = g.G.neighbors(u)
	nghs.append(u)
	H = g.G.subgraph(nghs)
	#nx.draw_spring(H,node_color=[float(H.degree(v)) for v in H],node_size=15,with_labels=True,cmap=plt.cm.Reds,)
	#plt.show()
	# prepare a circular layout of nodes
	pos = nx.spring_layout(H)
	#pos = nx.shell_layout(H)
	# define the color to select from the color map
	# as n numbers evenly spaced between color map limits
	node_color = map(int, np.linspace(0, 255, len(nghs)))
	# draw the nodes, specifying the color map and the list of color
	nx.draw_networkx_nodes(H, pos,node_color=node_color, cmap=plt.cm.hsv)
	# add the labels inside the nodes
	nx.draw_networkx_labels(H, pos)
	# draw the edges, using alpha parameter to make them lighter
	nx.draw_networkx_edges(H, pos, alpha=0.4)
	# turn off axis elements
	plt.axis('off')
	plt.show()
#this part is for plotting the connectivity between a node and its neighbors in order to study the strength of ties    
if(0):
	g = WCG(2048)
	g.build_csv("/Users/kazemjahanbakhsh/Downloads/export_bar.csv", 0, "\t")
	print nx.number_of_nodes(g.G)
	#t = g.spread_wties(True)
	#t = g.spread_Ngh(451, False)
	#print t
	#not_done = [54380, 54381, 54382, 54383, 54384, 54385, 54386, 54387, 54388]
	#not_done =[52851, 59490, 63321, 63483]
	#not_done = [60100, 60099, 62496]
	#not_done = [1, 2, 20, 24, 27]
	not_done = [62390]
	for u_ in not_done:
		#first level neighbors
		nghs_l1 = g.G.neighbors(u_)
		#print u_
		#print nghs_l1
#		nghs = g.G.neighbors(u_)
#		for v_ in nghs:
#			#print v_
#			nghs_l2 = g.G.neighbors(v_)
#			#print nghs_l2
#			for w_ in nghs_l2:
#				if w_ not in nghs_l1 and w_ != u_:
#					#print w_
#					nghs_l1.append(w_)
		nghs_l1.append(u_)
		H = g.G.subgraph(nghs_l1)
		#nx.draw_spring(H,node_color=[float(H.degree(v)) for v in H],node_size=10,with_labels=True,cmap=plt.cm.Reds,)
		#plt.show()
		g.plot_graph(H)
if(1):
	g = WCG(2048)
	G = nx.Graph()
	f = open('/Users/kazemjahanbakhsh/Downloads/cg.txt', 'r')
	for line in f:
		nodes = line.rstrip().split(',')
		G.add_edge(int(nodes[0]), int(nodes[1]))
	g.plot_graph(G)
#let's read all 1-whiskers and draw them from the file
if(0):
	#let's read the 1-whiskers file
	start_ = 0
	line_ = -1
	one_whiskers = []
	f = open('/Users/kazemjahanbakhsh/Downloads/out_whiskers', 'r')
	for line in f:
		if string.find(line, "no of components after removing bridges") >= 0:
			print line
			start_ = 1
			line_ = 0
		if start_ and line_ > 0 and (line_ % 2 == 0):
			T1 = line.split('[')[1].split(']')[0].split(',')
			#print T1
			comp_ = [int(x) for x in T1]
			one_whiskers.append(comp_)
		if line_ >= 0:
			line_ += 1
	g = WCG(2048)
	g.build_csv("/Users/kazemjahanbakhsh/Downloads/facebook-links.txt", 0, "\t")
	for comp_ in one_whiskers:
		print comp_
		#H = g.G.subgraph(comp_)
		new_comp = []
		for u_ in comp_:
			nghs_l1 = g.G.neighbors(u_)
			for v_ in nghs_l1:
				if v_ not in comp_:
					src_ = v_
				if v_ not in new_comp:
					new_comp.append(v_)
#				nghs_l2 = g.G.neighbors(v_)
#				for w_ in nghs_l2:
#					if w_ not in new_comp:
#						new_comp.append(w_)			
		print new_comp
		H = g.G.subgraph(new_comp)
		T = nx.bfs_tree(H, src_)
		print set(new_comp) - set(comp_)
		print T.edges()
#		bet = nx.edge_betweenness_centrality(H)
#		K = nx.Graph()
#		for u_ in comp_:
#			u_max = 0
#			u_ngh = -1
#			for k, v in bet.iteritems():
#				if u_ in k and v > u_max:
#					if k[0] != u_:
#						u_ngh = k[0]
#					else:
#						u_ngh = k[1]
#			K.add_edge(u_,u_ngh)  
		nx.draw_spring(H,node_color=[float(H.degree(v)) for v in H],node_size=10,with_labels=True,cmap=plt.cm.Reds,)
		plt.show()
		g.plot_graph(T)	
#		nx.draw_spring(K,node_color=[float(K.degree(v)) for v in K],node_size=10,with_labels=True,cmap=plt.cm.Reds,)
#		plt.show()
#let's count the number of Type I and II 1-whiskers
if(0):
	type_I = 0
	type_II = 0
	max_type_I = 0
	max_type_II = 0
	type_I_len = []
	type_II_len = []
	#let's read the 1-whiskers file
	start_ = 0
	line_ = -1
	one_whiskers = []
	f = open('/Users/kazemjahanbakhsh/Downloads/out_whiskers', 'r')
	for line in f:
		if string.find(line, "no of components after removing bridges") >= 0:
			print line
			start_ = 1
			line_ = 0
		if start_ and line_ > 0 and (line_ % 2 == 0):
			T1 = line.split('[')[1].split(']')[0].split(',')
			#print T1
			comp_ = [int(x) for x in T1]
			one_whiskers.append(comp_)
		if line_ >= 0:
			line_ += 1
	g = WCG(2048)
	g.build_csv("/Users/kazemjahanbakhsh/Downloads/facebook-links.txt", 0, "\t")
	for comp_ in one_whiskers:
		if(len(comp_) > 2):
			plot_flag_ = False
			print comp_
			new_comp = []
			for u_ in comp_:
				nghs_l1 = g.G.neighbors(u_)
				for v_ in nghs_l1:
					if v_ not in comp_:
						src_ = v_
					if v_ not in new_comp:
						new_comp.append(v_)
			for u_ in list(set(g.G.neighbors(src_)) & set(comp_)):
				print "(u,v) =(%d,%d) degree = (%d,%d) " % (u_, src_, len(g.G.neighbors(u_)), len(g.G.neighbors(src_)))
				if(len(g.G.neighbors(u_)) == 2):
					type_I += 1
					if(len(comp_) > max_type_I):
						max_type_I = len(comp_)
					type_I_len.append(len(comp_))
					if(len(comp_) >= 4):
						plot_flag_ = True
				else:
					type_II += 1
					if(len(comp_) > max_type_II):
						max_type_II = len(comp_)
					type_II_len.append(len(comp_))
	#		print new_comp
			if(plot_flag_):
				H = g.G.subgraph(new_comp)
				T = nx.bfs_tree(H, src_)
				nx.draw_spring(H,node_color=[float(H.degree(v)) for v in H],node_size=10,with_labels=True,cmap=plt.cm.Reds,)
				plt.show()
				g.plot_graph(T)	
	print "(type I, type II) = (%d,%d)" % (type_I, type_II)
	print "max(type I, type II) = (%d,%d)" % (max_type_I, max_type_II)
	print type_I_len
	print type_II_len
#let's read the centralities and examine their properties
if(0):
	#let's find the map 
	f1 = open('/Users/kazemjahanbakhsh/Downloads/core.csv', 'r')
	cnt = 0
	nodes_no = -1
	edges_no = 0
	g = WCG(2048)
	fmap = {}
	rmap = {}
	id_ = 0
	for line in f1:
		if cnt == 0:
			nodes_no = int(line.split('\n')[0])
			print "nodes no: %d" % nodes_no
		elif cnt == 1:
			edges_no = int(line.split('\n')[0])
			print "edge no: %d" % edges_no
		elif cnt <= nodes_no + 1:
			#build the map
			fmap[int(line.split(',')[0])] = id_	#mapping graph nodes ids to centrality ids
			rmap[id_] = int(line.split(',')[0])  #mapping centrality ids to graph nodes ids
			id_ += 1
			print "(%d --> %d)" % (int(line.split(',')[0]), id_)
		else:
			g.G.add_edge(int(line.split(',')[0]), int(line.split(',')[1].split('\n')[0]), weight=float(1))
		cnt += 1
	print "finish reading G"
	f2 = open('/Users/kazemjahanbakhsh/Downloads/centralities.txt', 'r')
	cnt = 0
	max_cent = 0.0
	high_cent = 0
	#we need this for histogram
	centralities = []
	#a dictionary for later queries
	cent_map = {}
	cent_2183 = []
	for line in f2:
		cnt += 1
		if(cnt >= 816887):
			curr_cent =float(line.split(': ')[1])
			#store centralities for plotting histogram
			centralities.append(curr_cent)
			id1 = int(line.split(': ')[0].split('(')[1].split(')')[0].split(',')[0])
			id2 = int(line.split(': ')[0].split('(')[1].split(')')[0].split(',')[1])
			pair = rmap[id1], rmap[id2]
			cent_map[pair] = curr_cent
			if int(line.split(': ')[0].split('(')[1].split(')')[0].split(',')[0]) == 2183:
				#let's store the distribution of edges adjacent to 2183
				cent_2183.append(curr_cent)
			if max_cent < curr_cent:
				#find the edge with highest betweenness
				max_cent = curr_cent
				max_pair = line.split(': ')[0]
			if curr_cent > 63391.0:
				#how many edges have high centrality
				high_cent += 1
	print max_pair
	for i in range(0,2):
		id_ = int(max_pair.split('(')[1].split(')')[0].split(',')[i])	#centrality id
		print "(%d --> %d)" % (id_, rmap[id_])
		nghs_ = g.G.neighbors(rmap[id_])
		print "degree: %d" % len(nghs_)
		H = g.G.subgraph(nghs_)
		#g.plot_graph(H)
	print max_cent
	print high_cent
	f1.close()
	f2.close()
	#plot the centrality histogram
	#norm_cent = [ x/(len(centralities)*1.0) for x in centralities]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	n, bins, rectangles = ax.hist(centralities, 100, normed=True)
	fig.canvas.draw()
	plt.show()
	cent_thr = [100, 500, 1000, 2000, 4000, 8000, 16000]
	for i in cent_thr:
		counter  = 0
		for cent_ in centralities:
			if cent_ <= i:
				counter += 1
		print "no of edges less than %d is %d" % (i, counter)
	#plt.hist(centralities, 100, normed=True)
	#plt.show()
	cent_2183.sort()
	print  cent_2183
	#norm_cent_2183 = [ x/(len(cent_2183)*1.0) for x in cent_2183]
	plt.hist(cent_2183, 100, normed=True)
	plt.show()
	print "start writing no of nodes and edges"
	h = open('/Users/kazemjahanbakhsh/Downloads/weighted_adj_cent.csv', 'w')
	num_nodes = g.G.number_of_nodes()
	value = ''
	value += str(num_nodes)
	value += '\n' 
	h.write(value)		
	num_edges = g.G.number_of_edges()		
	value = ''
	value += str(num_edges)
	value += '\n'
	h.write(value)
	print "start writing nodes ids and degrees"		
	nodes_ = g.G.nodes()
	for n_ in nodes_:
		value = ''
		value += str(n_)
		value += ','
		value += str(g.G.degree(n_))
		value += '\n' 
		h.write(value)
	print "start storing adj lists"
	for n_ in nodes_:
		#print n_
		nghs_ = g.G.neighbors(n_)
		pair_list = []
		for ngh in nghs_:
			#let's create the list of pairs (node_id, centrality)
			if n_ < ngh:
				node_pair = n_, ngh
			else:
				node_pair = ngh, n_
			pair_ = (ngh, cent_map[node_pair])
			pair_list.append(pair_)
		#sort the list of weights
		pair_list.sort(pair_inv_compare)
		#now store the sorted list
		for pair_ in pair_list:
			value = ''
			value = str(n_)
			value += ','
			value += str(pair_[0])
			value += ','
			value += str(pair_[1])
			value += '\n'
			h.write(value)
	h.close()
	print "finish writing centrality adjacency list"
	#let's read the 1-whiskers file
	start_ = 0
	line_ = -1
	one_whiskers = []
	#read the 1-whiskers and store them in a list
	f3 = open('/Users/kazemjahanbakhsh/Downloads/out_whiskers', 'r')
	for line in f3:
		if string.find(line, "no of components after removing bridges") >= 0:
			#print line
			start_ = 1
			line_ = 0
		if start_ and line_ > 0 and (line_ % 2 == 0):
			T1 = line.split('[')[1].split(']')[0].split(',')
			#print T1
			comp_ = [int(x) for x in T1]
			one_whiskers.append(comp_)
		if line_ >= 0:
			line_ += 1
	f3.close()
	#let's test our cent dictionary
	pair = 2184, 2332
	print "test dict: %f" % cent_map[pair]
	print "Length = ",len(cent_map)
	print "no of whiskers with size >= 2 = ",len(one_whiskers) 
	one_whisker_edges = 0
	#let's find the distribution of centrality of edges inside 1-whiskers
	one_whiskers_cent = []
	for comp in one_whiskers:
		H = g.G.subgraph(comp)
		one_whisker_edges += H.number_of_edges()
		for e in H.edges():
			pair = min(e[0], e[1]), max(e[0],e[1])
			one_whiskers_cent.append(cent_map[pair])
	print "no edges in whiskers: ", one_whisker_edges
	#let's plot the whiskers centralities
	plt.hist(one_whiskers_cent, 100)
	#plt.show()
	#print nx.diameter(g.G)
	#T = nx.bfs_tree(g.G, 2184)
	whisk = [60512, 60513, 60514, 60515, 60516, 60517, 60518, 60519, 60520, 60508, 60509, 60510, 60511, 63008, 63687, 54988, 54989, 54990, 54991, 55065, 55066, 63356, 63357, 63007,54380, 54381, 54382, 54383, 54384, 54385, 54386, 54387, 54388, 35581, 62496, 60099, 60100, 45811, 45812, 45813, 50652, 61054, 61055,48870, 48871, 48872, 49773, 49774, 50609, 50610, 50611, 50612,57953, 57801, 57802, 57946, 53501, 53502, 53503]
	for i in whisk:
		print i
		bfs = g.BreadthFirstLevels(i,100)
		for n in bfs:
			val_ = n.values()
	comps = [[62496, 60099, 60100, 45811, 45812, 45813, 50652, 61054, 61055],[57953, 57801, 57802, 57946, 53501, 53502, 53503],[48870, 48871, 48872, 49773, 49774, 50609, 50610, 50611, 50612], [62496, 60099, 60100, 45811, 45812, 45813, 50652, 61054, 61055]]
	for comp in comps:
		H = g.G.subgraph(comp)
		g.plot_graph(H)
	
	print(nx.is_connected(g.G))
	print g.G.number_of_edges()
	print g.G.number_of_nodes()
	#print T.number_of_edges()
	#print T.number_of_nodes()
	#T_undir = T.to_undirected()
	#print(nx.is_connected(T_undir))
	#print nx.diameter(T_undir)
#let's construct the epinion graph
if(0):
	#let's construct the epinion social graph
	g = WCG(2048)
	g.build_csv("/Users/kazemjahanbakhsh/Downloads/data/soc-Epinions1.txt", 0, "\t")
	#remove bridges
	#g.remove_bridges("/Users/kazemjahanbakhsh/Downloads/epinion_bridges.txt",0,",")
	#write the core of the graph into a file
	#g.export_G_to_csv(1)
	#find 1-whiskers
	g.findWhiskers("/Users/kazemjahanbakhsh/Downloads/epinion_bridges.txt","/Users/kazemjahanbakhsh/Downloads/data/soc-Epinions1.txt",0,"\t")
#	H = g.G.subgraph([75790, 75791, 75792, 75793, 75794, 75795, 75796, 75797, 75798, 75799, 75800, 75801, 75802, 75803, 75804, 75805, 75806, 75813, 75814, 75815, 74456, 75628, 75629, 75630, 75631, 75632, 75763, 75764, 75765, 75766, 75767, 75768, 75769, 75770, 75771])
#	g.plot_graph(H)
#	H = g.G.subgraph([70824, 67599, 67600, 67601, 67602, 47541, 47542, 47543, 9883, 71548])
#	g.plot_graph(H)
#	H = g.G.subgraph([75616, 75617, 75618, 75619, 74411, 75762, 75613, 75614, 75615])
#	g.plot_graph(H)
#	H = g.G.subgraph([75872, 75873, 75879, 75880, 75881, 75882, 75883, 75829, 75865])
#	g.plot_graph(H)
#	H = g.G.subgraph([28234, 61963, 61964, 61965, 61966, 61967, 61968, 61962])
#	g.plot_graph(H)

#let's check if the spnner constructed by taking the max centarlity is connected or not
if(0):
	g = WCG(2048)
	reader = csv.reader(open('/Users/kazemjahanbakhsh/Downloads/weighted_adj_cent.csv'), delimiter=',')
	cnt_1 = 1
	cnt_2 = 0
	node_u = -1
	node_v = -1
	max_cent = 0
	for line in reader:
		if cnt_1 >= 63395:
			if node_u == -1:
				node_u = int(line[0])
				node_v = int(line[1])
				max_cent = float(line[2])
				g.G.add_edge(node_u,node_v,weight=max_cent)
			elif node_u != int(line[0]):
#				print "(%d,%d) weight = %f" % (node_u, node_v, max_cent)
				node_u = int(line[0])
				node_v = int(line[1])
				max_cent = float(line[2])
				g.G.add_edge(node_u,node_v,weight=max_cent)
#				cnt_2 += 1
#				if cnt_2 == 5:
#					break
		else:
			cnt_1 += 1
	print g.G.number_of_nodes()
	print g.G.number_of_edges()
	print nx.number_connected_components(g.G)
	comps = nx.connected_components(g.G)
	max_comp = 0
	for comp in comps:
		if len(comp) > max_comp:
			max_comp = len(comp)
		if len(comp) == 2:
			print comp
	print max_comp
#let's check if the spanner constructed by taking all edges > threshold is connected or not
if(0):
	g = WCG(2048)
	reader = csv.reader(open('/Users/kazemjahanbakhsh/Downloads/weighted_adj_cent.csv'), delimiter=',')
	cnt_1 = 1
	thresh_ = 0.9
	exponent = 0.2
	#gspars
#	for line in reader:
#		if cnt_1 >= 63395:
#			node_u = int(line[0])
#			node_v = int(line[1])
#			strength = float(line[2])			
#			if strength <= thresh_:
#				g.G.add_edge(node_u,node_v,weight=strength)
#		else:
#			cnt_1 += 1
	#lsparse
	prev_node = 1
	list_ = []
	test = 0
	for line in reader:
		if cnt_1 >= 63395:
			node_u = int(line[0])
			node_v = int(line[1])
			strength = float(line[2])
			if(node_u == prev_node):		
				list_.append(node_v)
			else:
				deg_ = len(list_)
				num_ = int(math.pow(deg_, exponent))
				#print "node: %d, deg: %d, num_ : %d" % (prev_node, deg_, num_)
				for i in range(0, num_):
					g.G.add_edge(prev_node,list_[i], weight=strength)
					#print "edge: %d --> %d" % (prev_node, list_[i])
#				test += 1
#				if test == 4:
#					print "done!"
#					break
				del list_[:]
				prev_node = node_u
				list_.append(node_v)
		else:
			cnt_1 += 1
	print "|V| = %d, |E| = %d, |comp| = %d" % (g.G.number_of_nodes(), g.G.number_of_edges(), nx.number_connected_components(g.G))
	
	comps = nx.connected_components(g.G)
	max_comp_len = 0
	comp_two = 0
	for comp in comps:
		if len(comp) > max_comp_len:
			max_comp_len = len(comp)
			max_comp = comp
		if len(comp) == 2:
			comp_two += 1
	print "|MAX COMP| = %d, |COMP TWO| = %d" % (max_comp_len, comp_two)
	H = g.G.subgraph(max_comp)
	print "|V_H| = %d, |E_H| = %d" % (H.number_of_nodes(), H.number_of_edges())
	print "H's diameter: %d" % nx.diameter(H)
#let's find those nodes that finish the last and study their components properties
if(0):
	#let's construct the graph
	g = WCG(2048)
	g.build_csv("/Users/kazemjahanbakhsh/Downloads/facebook-links.txt", 0, "\t")
	#let's read the 1-whiskers file
	start_ = 0
	line_ = -1
	one_whiskers = []
	#read the 1-whiskers and store them in a list
	f = open('/Users/kazemjahanbakhsh/Downloads/out_whiskers', 'r')
	for line in f:
		if string.find(line, "no of components after removing bridges") >= 0:
			#print line
			start_ = 1
			line_ = 0
		if start_ and line_ > 0 and (line_ % 2 == 0):
			T1 = line.split('[')[1].split(']')[0].split(',')
			#print T1
			comp_ = [int(x) for x in T1]
			one_whiskers.append(comp_)
		if line_ >= 0:
			line_ += 1
	f.close()
	#read the last nodes and see if they are in 1-whisker components
	read_flag_ = False
	line_ = -1
	not_done_list = []
	f = open('/Users/kazemjahanbakhsh/Downloads/not_done_nodes_run_2', 'r')
	for line in f:
		if string.find(line, "Not done yet nodes:") >= 0:
			#print line
			read_flag_ = True
			line_ = 0
		if read_flag_ and line_ == 1:
			read_flag_ = False
			line_ = -1
			T1 = line.split(',')
			T1.remove(' \n')
			#print T1
			not_done_ = [int(x) for x in T1]
			not_done_list.append(not_done_)
		if line_ == 0:
			line_ += 1
#	print "partial"
#	print len(not_done_list)
#	print not_done_list[len(not_done_list) - 10:]
#	print list(set(not_done_list[14]) - set(not_done_list[15]))
	f.close()
	list_whiskers = []
	for ii in range(0, len(not_done_list)):
		#let's find nodes that are done by now
		#done_nodes_ = list(set(not_done_list[ii - 1]) - set(not_done_list[ii]))
		cnt_ = 0
		list_nodes = []
		list_nodes = not_done_list[ii]
		for node_ in list_nodes:
			#let's see if there is any node that dont belong to 1-whiskers
			belong_flag = False
			for whisker in one_whiskers:
				if node_ in whisker:
					belong_flag = True
					if whisker not in list_whiskers:
						#print "one_whisker"
						#print whisker
						list_whiskers.append(whisker)
					break
			if belong_flag == False:
				#let's check if this node is on the other side of the bridge
				for whisker in one_whiskers:
					for n_ in whisker:
						if node_ in g.G.neighbors(n_):
							belong_flag = True
							break
					if belong_flag:
						break
				if belong_flag == False:
					#print "node: %d is not in 1-whiskers" % node_
					cnt_ += 1
		print "%d: nodes not done: %d & not in 1-whiskers: %d --> percentage: %f" % (ii+33, len(list_nodes), cnt_, float(cnt_)/float(len(list_nodes)))
	#let's identify the not done 1-whiskers
	print "list whiskers:"
	print list_whiskers
	print len(list_whiskers)
	#let's process the size of 1-whiskers
	whiskers_len = {}
	for comp_ in list_whiskers:
		if len(comp_) in whiskers_len.keys():
			whiskers_len[len(comp_)] += 1
		else:
			whiskers_len[len(comp_)] = 1
	print whiskers_len
	core_deg = []
	whisker_deg = []
	for comp_ in list_whiskers:
		if (len(comp_) > 5):
			new_comp = []
			src_ = -1
			for u_ in comp_:
				nghs_l1 = g.G.neighbors(u_)
				for v_ in nghs_l1:
					if v_ not in comp_:
						#store the nodes id that are across the bridge
						src_ = v_
						across_src_ = u_
					if v_ not in new_comp:
						new_comp.append(v_)
			#find the neighbors of the node in the core that is connected to 1-whiskers through a bridge 
			if(src_ != -1):
				#let's extract the neighbors of the node in the core
				nghs_src = g.G.neighbors(src_)
				for v_ in nghs_src:
					if v_ not in new_comp:
						new_comp.append(v_)
				#let's find out the type of 1-whisker
				print "size(whisker): %d: degree of node in the core: %d & node across the bridge in 1-whisker: %d" % (len(comp_), g.G.degree(src_), g.G.degree(across_src_))
				#let's compute the degree distribution of nodes inside the 1-whisker and core
				whisker_deg.append(g.G.degree(across_src_))
				core_deg.append(g.G.degree(src_))
			#print new_comp
			H = g.G.subgraph(new_comp)
			#nx.draw_spring(H,node_color=[float(H.degree(v)) for v in H],node_size=10,with_labels=True,cmap=plt.cm.Reds,)
			#plt.show()
			g.plot_graph(H)	
	#print core_deg
	#print whisker_deg
#	plt.hist(core_deg, 100)
#	plt.show()
#	plt.hist(whisker_deg, 10)
#	plt.show()
if(0):
	num_ = 30
	g = WCG(num_)
	g.G.add_edge(0,1,weight=float(1))
	for i in range(2, num_):
		g.G.add_edge(0, i, weight=float(1))
		g.G.add_edge(1, i, weight=float(1))
    
	cnt_0 = 0
	cnt_1 = 0
	times = 1000
	for i in range(times):
	    dic = g.spread_Censor(False)
	    if dic[0].count(1) == 1:
	        cnt_0 = cnt_0 + 1
	    if dic[1].count(0) == 1:
	        cnt_1 = cnt_1 + 1
	print "(%f,%f)" % (float(cnt_0)/float(times),float(cnt_1)/float(times))
#let's plot centralities histogram
if(0):
	f = open('/Users/kazemjahanbakhsh/Downloads/centralities.txt', 'r')
	#we need this for histogram
	centralities = []
	#a dictionary for later queries
	cnt = 0
	for line in f:
		cnt += 1
		if(cnt >= 816887):
			curr_cent = float(line.split(': ')[1])
			#store centralities for plotting histogram
			centralities.append(curr_cent)
	f.close()
	#plot the centrality histogram
	#norm_cent = [ x/(len(centralities)*1.0) for x in centralities]
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#n, bins, rectangles = ax.hist(centralities, 100, normed=True)
	#fig.canvas.draw()
	#plt.show()
	#let's plot the histogram of centralities
	print plt.hist(centralities, 100)
	plt.show()
if(0):
	H = nx.read_gml('/Users/kazemjahanbakhsh/Downloads/data/netsci/netscience.gml')
	print H.number_of_nodes()
	print H.number_of_edges()
	print H.degree(19)
	num_ = H.number_of_nodes()
	#g = WCG(num_)
	#g.G = H
	#for edge_ in H.edges():
		#g.G.add_edge(edge_[0] - 1, edge_[1] - 1, weight=float(1))
		
	print nx.betweenness_centrality(H)
	comps = nx.connected_components(H)
	K = H.subgraph(comps[0])
	mapping = {}
	start_ = 0
	for node in K.nodes():
		mapping[node] = start_
		start_ = start_ + 1
	num_ = K.number_of_nodes()
	g = WCG(num_)
	for edge_ in K.edges():
		g.G.add_edge(mapping[edge_[0]], mapping[edge_[1]], weight=float(1))
	print K.number_of_edges()
	print K.number_of_nodes()
	nx.draw_spring(g.G,node_color=[float(g.G.degree(v)) for v in g.G],node_size=10,with_labels=True,cmap=plt.cm.Reds,)
	plt.show()
	times = -1
	r1 = 0
	r2 = 0
	for i in range(times + 1):
		r1 = r1 + g.spread_Censor(False)
		r2 = r2 + g.spread_RW(False)
	
	print float(r1)/float(times)
	print float(r2)/float(times)
	
if(0):
	g = WCG(1)
	#l1 = [1, 5, 6, 8, 21, 34, 19, 17, 2]
	#l2 = [11, 2, 5, 17, 19, 43, 68]
	l1 = [1, 2, 4]
	l2 = [2, 5]
	g.test_minhash(l1, l2)
if(0):
	g = WCG(1800)
	g.build_tri()
#	g.calc_probs(0, 100)
	t = g.spread_RW(False)
	print t
####export end
r = 0
k = 0
for i in range(0,k):
     g = WCG(2048)    
     g.build_BarbellG(2)
#     g.build_Clique()
#    g.build_watts_strogatz(3, 0.01)
#    g.build_csv("/home/kazem/Data/UCI/karate.txt", 1, " ")
#    g.build_csv("/home/kazem/weakc/facebook-links.txt", 0, "\t")
#    g.build_csv("soc-LiveJournal1.txt", 0, "\t")
#    g.build_hierarichal_watts(2, 4, 1.5, 10.0)
#     t = g.spread_RW()
     t = g.spread_Censor(False)
#    print t
     r += t
#if k != 0:
#    avg_ = r/k
#print "the average round#: %f" % avg_
#print "no of components after running BFS on G wo bridges: %d" % nx.number_connected_components(g.G)
#comps = nx.connected_components(g.G)
#for comp in comps:
#	print len(comp)
##########computing centrality of nodes
#print "just created the graph"
#centrality = nx.betweenness_centrality(g.G)
#f_cent = open('/home/kazem/weakc/centrality.txt', 'w')
#for u, cent_val in centrality.iteritems():
#    value = ''
#    value += str(u)
#    value += ":"
#    value += str(cent_val)
#    value += '\n' 
#    f_cent.write(value)      


#print nx.betweenness_centrality(g.G)
#nx.draw_spring(H,node_color=[float(H.degree(v)) for v in H],node_size=10,with_labels=False,cmap=plt.cm.Reds,)
#plt.savefig("/home/kazem/plots/whiskers.eps")
#plt.show()
#g = WCG(100)
#g.build_BarbellG(2)
#nx.draw_spring(g.G,node_color=[float(g.G.degree(v)) for v in g.G],node_size=10,with_labels=True,cmap=plt.cm.Reds,)
#plt.show()
#g.rw(100)
#print g.G.nodes()
#print g.G.edges()
#print "the avg: %f" % r
#g = WCG(0)
#g.build_csv()
#print "no of nodes: %d" % g.G.number_of_nodes()
#print "no of edges: %d" % g.G.number_of_edges()
#if (g.G.has_edge(1, 2) == True):
#    print "there is an edge from 1 to 2!"
#if (g.G.has_edge(2, 1) == True):
#    print "there is an edge from 2 to 1!"

#testing tree
#root = Tree()
#root.id = 0
#root.left = Tree()
#root.left.id = 1
#root.right = Tree()
#root.right.id = 2
#
#root.preorder()
#print nx.degree(g.G).values()
#nx.draw_spring(g.G,node_color=[float(g.G.degree(v)) for v in g.G],node_size=1,with_labels=True,cmap=plt.cm.Reds,)
#plt.show()
