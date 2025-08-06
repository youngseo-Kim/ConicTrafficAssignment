import pandas as pd
import numpy as np
import networkx as nx
import argparse



""" 
parameters
"""
# subway_speed_factor = 0.5
# walk_speed_factor = 12
# bus_speed_factor = 4

subway_speed_factor = 1
walk_speed_factor = 1
bus_speed_factor = 1
large_C = 100000


# Set up argument parser
parser = argparse.ArgumentParser(description="Script to run routing experiments.")
parser.add_argument("--data_type", type=str, default="Munich", help="Dataset to use (e.g., Munich, GoldCoast, etc.)")
parser.add_argument("--n_routes", type=int, default=3, help="Number of routes to consider per OD pair")

# Parse arguments
args = parser.parse_args()

# Assign variables from parsed arguments
data_type = args.data_type
n_routes = args.n_routes

# Print to verify
print(f"data_type = {data_type}")
print(f"n_routes = {n_routes}")

multi_mode = True 

network_df = pd.read_csv("../data/{}/{}_net.txt".format(data_type, data_type), sep='\t', comment=';')
# node_df = pd.read_csv("../data/{}/{}_node.txt".format(data_type, data_type), sep='\t', comment=';')
od_df = pd.read_csv("../data/{}/{}_od.csv".format(data_type, data_type))
od_df = od_df[od_df['Ton'] > 0] # Select positive values only 

if data_type == "Chicago":
    network_df = network_df[['init_node', 'term_node', 'capacity', 'free_flow_time', 'free_flow_time_save', 'b',
       'power', 'speed', 'toll', 'link_type']] # Chicago
    network_df['length'] = network_df['free_flow_time'] # Chicago

elif data_type == "GoldCoast":
    network_df = network_df[['init_node', 'term_node', 'capacity', 'length', 'free_flow_time', 'b',
       'power', 'speed ', 'critical_speed', 'lanes']] 

elif data_type == "Sydney":
    network_df = network_df[['init_node', 'term_node', 'capacity', 'length', 'free_flow_time', 'b',
       'power', 'speed', 'critical_speed', 'lanes']] 

else:
    network_df = network_df[['init_node', 'term_node', 'capacity', 'length', 'free_flow_time', 'b',
        'power', 'speed', 'toll', 'link_type']]

network_df[['init_node', 'term_node']] = network_df[['init_node', 'term_node']].astype(int)

# node_df = node_df[['Node', 'X', 'Y']]

# print("Number of nodes:", len(node_df))
print("Number of links:", len(network_df))
print("Number of positive od pairs:", len(od_df))

# Find unique origins and destinations
origins = od_df['O'].unique()
destins = od_df['D'].unique()

od = []
for od_pair_index in range(len(od_df)):
    i,j = od_df['O'].iloc[od_pair_index], od_df['D'].iloc[od_pair_index]
    od.append((i,j))

# road_link = [(int(row['init_node']), int(row['term_node']), row['free_flow_time']) for _, row in network_df.iterrows() if row['free_flow_time'] == 'inf' (int(row['init_node']), int(row['term_node']), large_C)]
road_link = [
    (int(row['init_node']), int(row['term_node']), large_C if row['free_flow_time'] == 'inf' else max(0, row['free_flow_time']))
    for _, row in network_df.iterrows()
]

if multi_mode == True:
    # transit_line = [(1, 3), (3, 12), (12, 13), (4, 11), (11, 14), (14, 23), (23, 24), (5, 9), (9, 10), (10, 15), (15, 22), (22, 21), (2, 6), (6, 8), (8, 16), (16, 17), (17, 19), (19, 20),
    #                 (3, 1), (12, 3), (13, 12), (11, 4), (14, 11), (23, 14), (24, 23), (9, 5), (10, 9), (15, 10), (22, 15), (21, 22), (6, 2), (8, 6), (16, 8), (17, 16), (19, 17), (20, 19)]  

    # transit_line = [(2, 6), (6, 8), (8, 16), (16, 17), (17, 19), (19, 20), (20, 21), (21, 24), (24, 13), (12, 11), (11, 10), (10, 16),
    #                 (6, 2), (8, 6), (16, 8), (17, 16), (19, 17), (20, 19), (21, 20), (24, 21), (13, 24), (11, 12), (10, 11), (16, 10)]  

    transit_line = []


    subway_link = [
        (int(row['init_node']), int(row['term_node']), row['free_flow_time']*subway_speed_factor) 
        if (int(row['init_node']), int(row['term_node'])) in transit_line 
        else (int(row['init_node']), int(row['term_node']), row['free_flow_time']*walk_speed_factor) 
        for _, row in network_df.iterrows()
    ]

    bus_link = [(int(row['init_node']), int(row['term_node']), row['free_flow_time']*bus_speed_factor) for _, row in network_df.iterrows()]


arcs = [(int(row['init_node']), int(row['term_node'])) for _, row in network_df.iterrows()]

OD_route = {}
OD_route_length = {}
OD_route_cost = {}



# def generate_route_sets_link_elimination(graph, source, target, num_routes):
#     route_sets = []
#     route_dict = {}
#     r_ix = 0
    
#     # while len(route_dict) < num_routes:
#     for i in range(num_routes):
#         path = nx.shortest_path(graph, source=source, target=target, weight='weight')
#         path = [int(p) for p in path]
#         if path not in route_sets:
#             route_sets.append(path)
#             route_dict[int(r_ix)] = list(path)
#             r_ix += 1 
        
#         if len(path) > 2:
#             edge_to_remove = (path[1], path[2])
#             original_weight = graph[edge_to_remove[0]][edge_to_remove[1]]['weight']
#             graph[edge_to_remove[0]][edge_to_remove[1]]['weight'] = float('inf')
            
#         else:
#             break
        
#     # # Reset graph weights for future usage
#     # nx.set_edge_attributes(graph, original_weight, 'weight')
    
#     return route_dict

# # link penalty approach
# def generate_route_sets_link_penalty(graph, source, target, num_routes, penalty_factor=1.05):
#     route_sets = []
#     route_dict = {}
#     r_ix = 0
#     # while len(route_dict) < num_routes:
#     for i in range(num_routes):
#         path = nx.shortest_path(graph, source=source, target=target, weight='weight')
#         path = [int(p) for p in path]
#         if path not in route_sets:
#             route_sets.append(path)
#             route_dict[int(r_ix)] = list(path)
#             r_ix += 1 
        
#         for j in range(len(path) - 1):
#             edge = (path[j], path[j+1])
#             graph[edge[0]][edge[1]]['weight'] *= penalty_factor 
            
#     # # Reset graph weights for future usage
#     # nx.set_edge_attributes(graph, 1, 'weight')

#     return route_dict


def generate_route_sets_link_penalty(graph, source, target, num_routes, penalty_factor=1.05):
    route_sets = []
    route_dict = {}
    route_lengths = {}
    graph_original = graph.copy()
    r_ix = 0
    for i in range(num_routes):
    # while len(route_dict) < num_routes:
        path = nx.shortest_path(graph, source=source, target=target, weight='weight')
        path_length = sum(graph_original[path[j]][path[j+1]]['weight'] for j in range(len(path) - 1))  # Calculate path length
        path = [int(p) for p in path]
        if path not in route_sets:
            route_sets.append(path)
            route_dict[int(r_ix)] = list(path)
            route_lengths[int(r_ix)] = path_length  # Store path length in route_lengths dict
            r_ix += 1 
        
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            graph[edge[0]][edge[1]]['weight'] *= penalty_factor 
    
    return route_dict, route_lengths  # Return both paths and their lengths

# Example usage:
# graph = nx.DiGraph()  # Assume graph has been created and populated with nodes and edges
# routes, lengths = generate_route_sets_link_penalty(graph, source, target, num_routes)



for (i,j) in od:

    if multi_mode == True: 
        G1 = nx.DiGraph()
        G1.add_weighted_edges_from(road_link)
        route_sets,route_lengths = generate_route_sets_link_penalty(G1, i, j, n_routes) 
        OD_route[(int(i),int(j)), "m1"] = route_sets 
        OD_route_length[(int(i),int(j)), "m1"] = route_lengths 

        G2 = nx.DiGraph()
        G2.add_weighted_edges_from(bus_link)
        route_sets,route_lengths = generate_route_sets_link_penalty(G2, i, j, 1) 
        OD_route[(int(i),int(j)), "m2"] = route_sets  # get the shortest path
        OD_route_length[(int(i),int(j)), "m2"] = route_lengths 

        
        G3 = nx.DiGraph()
        G3.add_weighted_edges_from(subway_link)
        route_sets,route_lengths = generate_route_sets_link_penalty(G3, i, j, 1) 
        OD_route[(int(i),int(j)), "m3"] = route_sets 
        OD_route_length[(int(i),int(j)), "m3"] = route_lengths 

    else: # if single mode 
        G = nx.DiGraph()
        G.add_weighted_edges_from(road_link)
        route_sets,route_lengths = generate_route_sets_link_penalty(G, i, j, n_routes) 
        # shortest_path = list(nx.shortest_path(G1, i, j, weight='weight'))
        OD_route[(int(i),int(j))] = route_sets 
        OD_route_length[(int(i),int(j))] = route_lengths 




def is_continuous_subsequence(my_tuple, my_list):
    tuple_length = len(my_tuple)
    list_length = len(my_list)
    
    for i in range(0, list_length - tuple_length + 1):
        if tuple(my_list[i:i+tuple_length]) == my_tuple:
            return True
    return False


import pickle
pd.to_pickle(OD_route, "../data/{}/OD_route_{}.pickle".format(data_type, n_routes))
pd.to_pickle(OD_route_length, "../data/{}/OD_route_length_{}.pickle".format(data_type, n_routes))

od_length = {"m1":[], "m2":[], "m3":[]}
for (i,j) in od:
    for m in ["m1", "m2", "m3"]:
        od_length[m].append(len(OD_route[(i,j), m]))

print("Average number of routes for road network: ", np.mean(od_length["m1"]))
print("Maximum number of routes for road network", np.max(od_length["m1"]))
print("Total number of routes for road network", np.sum(od_length["m1"]))


print("Average number of routes for bus network: ", np.mean(od_length["m2"]))
print("Maximum number of routes for bus network", np.max(od_length["m2"]))
print("Total number of routes for bus network", np.sum(od_length["m2"]))

print("Average number of routes for subway network: ", np.mean(od_length["m3"]))
print("Maximum number of routes for subway network", np.max(od_length["m3"]))
print("Total number of routes for subway network", np.sum(od_length["m3"]))
