# Note that this is the reduced version. We don't include experiments with M_set.

using CSV
using DataFrames
using LinearAlgebra
using Pandas: read_pickle
using JuMP
using Ipopt
using MosekTools
using SCS
using ECOS
using PyCall
using JSON

m_set = ["m1", "m2", "m3"]  # Alternative m

theta_dest = 1 # scaling parameters
theta_mode = 1
theta_route = 1
tau_M = Dict("M1" => 1, "M2" => 0.5)
fuel_cost = 0.5 # $/km 
value_of_time = 20 # $/hr


# Units
#The units of free flow travel time are often treated as minutes. so that given unit-less link lengths are treated as km.

dataset = length(ARGS) >= 1 ? ARGS[1] : "SiouxFalls"  # default is SiouxFalls network if not provided
solver = length(ARGS) >= 2 ? ARGS[2] : "MOSEK"        # default is MOSEK if not provided
Stage = length(ARGS) >= 3 ? ARGS[3] : "first"        
perturb = length(ARGS) >= 4 ? ARGS[4] : "false"        
transit = length(ARGS) >= 5 ? ARGS[5] : "false" # default is that the transit network is provided       
C = length(ARGS) >= 6 ? ARGS[6] : "1" # scaling factor   
scale_lambda = length(ARGS) >= 7 ? ARGS[7] : "1" # scaling factor   
entropy_cones = length(ARGS) >= 8 ? ARGS[8] : "exponential" # "relative" or "relative_ver2" 
beckmann = length(ARGS) >= 9 ? ARGS[9] : "power" # "power" or "socp" 
objective = length(ARGS) >= 10 ? ARGS[10] : "original" # or "beckmann" or "original"  
max_n_routes = length(ARGS) >= 11 ? ARGS[11] : "3" # or "beckmann" or "entropy"  



C = parse(Int, C)
scale_lambda = parse(Float64, scale_lambda)

# Print the values to verify
println("Dataset: $dataset")
println("Solver: $solver")
println("Stage: $Stage")
println("Whether data is perturb: $perturb")
println("Whether transit network is provided: $transit")
println("Scale parameter C: $C")
println("Scale parameter Lambda: $scale_lambda")
println("Implementation for entropy cones: $entropy_cones")
println("Implementation for Beckmann: $beckmann")
println("Objective function: $objective")

network_df = CSV.read(joinpath("..", "data", dataset, dataset * "_net.txt"), DataFrame, delim='\t')
network_df = select(network_df, Not(:Column1))
network_df = dropmissing(network_df, :free_flow_time) # drop rows with missing values in the free flow time column
network_df = network_df[isfinite.(network_df.free_flow_time), :]  # keep only rows where free_flow_time is finite

if perturb == "true"
    od_df = CSV.read(joinpath("..", "data", dataset, dataset * "_od_dist2.csv"), DataFrame)
else
    od_df = CSV.read(joinpath("..", "data", dataset, dataset * "_od.csv"), DataFrame)
end

od_df = filter(:Ton => >(20.0), od_df) # remove all the ods pair less than 20 demand 
println("Total number of OD pairs: ", nrow(od_df))

ods = [(row.O, row.D) for row in eachrow(od_df)] 
if dataset == "Chicago"
    road_link = Dict((row.init_node, row.term_node) => row.free_flow_time for row in eachrow(network_df))
else
    road_link = Dict((row.init_node, row.term_node) => row.length for row in eachrow(network_df))
end

A = [(row.init_node, row.term_node) for row in eachrow(network_df)] #arcs

O = unique(od_df[!, :O])
D = unique(od_df[!, :D])

alpha = 0.15
beta = 4
# convert hr to minutes
t0_am = Dict(((row.init_node, row.term_node), m) => row.free_flow_time for row in eachrow(network_df) for m in m_set)

d_od = Dict((row.O, row.D) => row.Ton for row in eachrow(od_df))
N = sum(d_od[od] for od in ods)


c_a = Dict((row.init_node, row.term_node) => row.capacity/C for row in eachrow(network_df))


max_c_a = maximum(values(c_a))




R = read_pickle("../data/$dataset/OD_route_$max_n_routes.pickle")
route_length = read_pickle("../data/$dataset/OD_route_length_$max_n_routes.pickle")

function print_total_routes(route_length, mode = "m1")
    total_routes = 0
    for ((i, j), m) in keys(route_length)
        if m == mode
            total_routes += length(route_length[(i, j), m])
        end
    end
    println("Total number of routes for road network (mode = $mode): ", total_routes)
end

print_total_routes(route_length, "m1")



py"""
def is_continuous_subsequence(my_tuple, my_list):
    tuple_length = len(my_tuple)
    list_length = len(my_list)
    
    for i in range(0, list_length - tuple_length + 1):
        if tuple(my_list[i:i+tuple_length]) == my_tuple:
            return True
    return False

"""


# Access the Python function as if it were a Julia function
is_continuous_subsequence = py"is_continuous_subsequence"



# TODO: Add V_ij and V_ijm. Currently we assume they are 0.


# V_ijmr = Dict()
# for od in ods
#     for r in keys(R[od, "m1"])
#         V_ijmr[od, "m1", r] = - route_length[od, "m1"][r]* fuel_cost # fuel cost is $1.2 per km 
#     end
# end

# V_ijm = Dict()
# for od in ods
#     for m in ["m2", "m3"]
#         V_ijm[od, m] = -2.5 + 10 # bus and subway cost is $2.5 per trip, ASC is set to 10
#     end
# end

println("Data loaded")



# ∑ = sum
# Model initialization

model_construction_time = @time begin
    
    println(Threads.nthreads())  # This function returns the number of threads Julia is currently using

    if solver == "MOSEK"
        FS = Model(Mosek.Optimizer)
        set_optimizer_attribute(FS, "MSK_IPAR_PRESOLVE_USE", 0)
        set_optimizer_attribute(FS, "MSK_IPAR_NUM_THREADS", Threads.nthreads())

        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1.0e-14)
        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_TOL_PFEAS", 1e-14)
        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_TOL_DFEAS", 1e-14)
        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-14)
        # set_optimizer_attribute(FS, "MSK_IPAR_INTPNT_SOLVE_FORM", dual) # choose to solve dual

        
    elseif solver == "ECOS"
        FS = Model(ECOS.Optimizer)
    elseif solver == "SCS"
        FS = Model(SCS.Optimizer)
        FS = Model(optimizer_with_attributes(SCS.Optimizer, "verbose" => true))
    elseif solver == "IPOPT"
        FS = Model(Ipopt.Optimizer)
    else
        raise("Solver not supported")
    end

    # C is the scaling parameter for both flow counts and the capacity



    # Variables for Beckmann equation
    # @variable(FS, f_am[a in A, m in m_set] >= 0) #Scaling: we never use f_am
    @variable(FS, f_am_bar[a in A, m in m_set] >= 0) #Scaling: we define f_am_bar = f_am/C # this is redundant
    @variable(FS, s_a[a in A] >= 0)
    @variable(FS, u_a[a in A] >= 0)
    @variable(FS, v_a[a in A] >= 0)

    @variable(FS, 0 <= p_ij[od in ods] <= 1)
    @variable(FS, 0 <= p_ijm[od in ods, m in m_set] <= 1)
    @variable(FS, 0 <= sum_p_ijm[od in ods, m in m_set] <= 1)
    @variable(FS, 0 <= sum_p_ijmr[od in ods, m in m_set, r in keys(R[od, m])] <= 1)
    @variable(FS, 0 <= p_ijmr[od in ods, m in m_set, r in keys(R[od, m])] <= 1)


    if entropy_cones == "relative_ver2"
        @variable(FS, 0 <= t_ij) 
        @variable(FS, 0 <= v_ijm)
        @variable(FS, 0 <= w_ijmr)
    elseif entropy_cones in ["relative", "exponential"]
        # auxiliary variable for exponential cone
        @variable(FS, 0 <= t_ij[od in ods] <= 0.3679) 
        @variable(FS, 0 <= v_ijm[od in ods, m in m_set] <= 0.3679)
        @variable(FS, 0 <= w_ijmr[od in ods, m in m_set, r in keys(R[od, m])] <= 0.3679)
    end

    # Note: changed it to inequality

    if (Stage == "first") 
        if entropy_cones == "relative_ver2"
            @constraint(FS, t_ij_lb, t_ij >= -sum(d_od[i,j]/N * log(d_od[i,j]/sum(d_od[i, jp] for jp in D if (i,jp) in ods)) for (i,j) in ods if d_od[i,j] > 0)) # observed constraints
        else
            @constraint(FS, t_ij_lb, sum(t_ij[od] for od in ods) >= -sum(d_od[i,j]/N * log(d_od[i,j]/sum(d_od[i, jp] for jp in D if (i,jp) in ods)) for (i,j) in ods if d_od[i,j] > 0)) # observed constraints
        end
    end

  
    @constraint(FS, [i in O], sum(p_ij[(i,j)] for j in D if (i,j) in ods) == sum(d_od[i, j] for j in D if (i,j) in ods)/N)
    @constraint(FS, [od in ods], p_ij[od] == sum(p_ijmr[od, m, r] for m in m_set for r in keys(R[od, m])))
    

    # TODO: write program for dual case
    if Stage == "second" # exponential cone 
        @objective(FS, Max, 
        - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N 
        - sum(t0_am[a,m] * f_am_bar[a, m] for m in ["m2", "m3"] for a in A) * C / N  
        + sum(t_ij[od] for od in ods) * theta_dest
        + sum(v_ijm[od, m] for od in ods for m in m_set)
        + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m])) 
        )

    elseif entropy_cones == "relative_ver2"
        @objective(FS, Max, 
        - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N 
        - sum(t0_am[a,m] * f_am_bar[a, m] for m in ["m2", "m3"] for a in A) * C / N 
        + t_ij
        + u_ijM 
        + v_ijm 
        + w_ijmr 
        )

    else

        if objective == "original"
            @objective(FS, Max, 
            - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N * scale_lambda
            - sum(t0_am[a,m] * f_am_bar[a, m] for m in ["m2", "m3"] for a in A) * C / N * scale_lambda  # need the right scale
            + sum(t_ij[od] for od in ods)
            + sum(sum(v_ijm[od, m] for od in ods for m in m_set) for m in m_set) 
            + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m])) 
            )

        # Only with Entropy functions
        elseif objective == "entropy"
            @objective(FS, Max, 
            + sum(t_ij[od] for od in ods)
            + sum(v_ijm[od, m] for od in ods for m in m_set)  
            + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m])) 
            )

        elseif objective == "beckmann"
            # # Only with Beckmann equation
            @objective(FS, Max, 
            - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N 
            - sum(t0_am[a,m] * f_am_bar[a, m] for m in ["m2", "m3"] for a in A) * C / N 
            )
        end
    end


    @constraint(FS, [a in A, m in ["m1"]], f_am_bar[a, m] * C == sum(p_ijmr[od, m, r] for od in ods for r in keys(R[od, m]) if is_continuous_subsequence(a,R[od, m][r])) * N ) #Scaling: use f_am_bar instead of f_am
    @constraint(FS, [a in A, m in ["m2", "m3"]], f_am_bar[a, m] * C == sum(p_ijmr[od, m, r] for od in ods for r in keys(R[od, m]) if is_continuous_subsequence(a,R[od, m][r])) * N)
   

    if entropy_cones == "exponential" # exponential cone 
        for od in ods
            (i,_) = od
            @constraint(FS, [t_ij[od], p_ij[od], sum(d_od[i, j] for j in D if (i,j) in ods)/N] in MOI.ExponentialCone())
        end

        # @constraint(FS, [od in ods, M in M_set], [u_ijM[od, M], p_ijM[od, M], sum(p_ijM[od, Mp] for Mp in M_set)] in MOI.ExponentialCone())

        @constraint(FS, [od in ods, m in m_set], sum_p_ijm[od, m] == sum(p_ijm[od, m] for m in m_set))
        @constraint(FS, [od in ods, m in m_set], [v_ijm[od, m], p_ijm[od, m], sum_p_ijm[od,m]] in MOI.ExponentialCone())

        @constraint(FS, [od in ods, m in m_set, r in keys(R[od, m])], sum_p_ijmr[od, m, r] == sum(p_ijmr[od, m, rp] for rp in keys(R[od, m])))
        @constraint(FS, [od in ods, m in m_set, r in keys(R[od, m])], [w_ijmr[od, m, r], p_ijmr[od, m, r], sum_p_ijmr[od, m, r]] in MOI.ExponentialCone())

    end

    if beckmann == "power"
        @constraint(FS, [a in A], [f_am_bar[a, "m1"], 1, s_a[a]] in MOI.PowerCone(1/5))
    elseif beckmann == "socp"
        @constraint(FS, [a in A], [2 * f_am_bar[a, "m1"], u_a[a] - 1, u_a[a] + 1] in SecondOrderCone())
        @constraint(FS, [a in A], [2 * u_a[a], v_a[a] - f_am_bar[a, "m1"], v_a[a] + f_am_bar[a, "m1"]] in SecondOrderCone())
        @constraint(FS, [a in A], [2 * v_a[a], s_a[a] - f_am_bar[a, "m1"], s_a[a] + f_am_bar[a, "m1"]] in SecondOrderCone())
    end


end


optimize!(FS)

# Check the status of the solution
status = termination_status(FS)
println("Termination status: $status")

println("Min p_ij: ", minimum(value.(p_ij)))
println("Max p_ij: ", maximum(value.(p_ij)))


f_am_solution = value.(f_am_bar * C)
p_ij_solution = value.(p_ij)
p_ijm_solution = value.(p_ijm)
# p_ijM_solution = value.(p_ijM)
p_ijmr_solution = value.(p_ijmr)

if Stage == "second"
    t_ij_dual = value.(theta_dest)
elseif Stage == "first"
    t_ij_dual = dual.(t_ij_lb) # FirstStage
end
# u_ijM_dual = Dict()  # Create a dictionary to store the dual values


# u_ijM_M1_dual = dual.(u_ijM_lb_M1)
# u_ijM_M2_dual = dual.(u_ijM_lb_M2)
# v_ijm_dual = dual.(v_ijm_lb)

obj_value = objective_value(FS)
println("Current best solution with objective value: $obj_value")

if Stage == "first"
    theta_dest = 1/(t_ij_dual + 1) # FirstStage
end
# theta_mode = 1/(v_ijm_dual + 1)

# tau_M1 = (u_ijM_M1_dual + 1)*theta_mode
# tau_M2 = (u_ijM_M2_dual + 1)*theta_mode
# tau_M = Dict("M1" => tau_M1, "M2" => tau_M2)

if Stage == "first"
    println("theta destination: ", theta_dest) # FirstStage
end
# println("theta mode: ", theta_mode)
# println("theta routing is assumed to be 1 WLOG")
# println("tau M1: ", tau_M1)
# println("tau M2: ", tau_M2)

py"""
import pandas as pd
# SiouxFalls
# transit_line = [(1, 3), (3, 12), (12, 13), (4, 11), (11, 14), (14, 23), (23, 24), (5, 9), (9, 10), (10, 15), (15, 22), (22, 21), (2, 6), (6, 8), (8, 16), (16, 17), (17, 19), (19, 20),
# (3, 1), (12, 3), (13, 12), (11, 4), (14, 11), (23, 14), (24, 23), (9, 5), (10, 9), (15, 10), (22, 15), (21, 22), (6, 2), (8, 6), (16, 8), (17, 16), (19, 17), (20, 19)]  


transit_line = [] 
subway_speed_factor = 1
walk_speed_factor = 1
bus_speed_factor = 1

# subway_speed_factor = 0.9
# walk_speed_factor = 0.9
# bus_speed_factor = 0.9


network_df = pd.read_csv("../data/{}/{}_net.txt".format($(dataset), $(dataset)), sep='\t', comment=';')
def bpr_func(m, a, flow):
    row = network_df[(network_df['init_node'] == a[0]) & (network_df['term_node'] == a[1])].iloc[0] # there must be one row
    t_0, alpha, capacity, beta = row['free_flow_time'], row['b'], row['capacity'], row['power']

    if m == "m1": # road
        return t_0 * (1 + alpha * (flow / capacity)**beta)
    if m == "m2": # bus
        return t_0 * bus_speed_factor
    elif m == "m3": # subway
        if a in transit_line:
            return t_0 * subway_speed_factor
        else:
            return t_0 * subway_speed_factor #walk_speed_factor

"""



bpr_func = py"bpr_func"



# Data structure to hold the output
f_am_dict = Dict()

print("optimal traffic flow \n")
for a in A
    for m in m_set
        key = (a, m)
        value = f_am_solution[a, m] #TODO: change this
        # Use string representation of key for JSON compatibility
        key_str = "($a, $m)"
        f_am_dict[key_str] = Dict("value" => value, "tt" => bpr_func(m, a, value))
    end
end



# Convert dictionary to JSON string
f_am_json = JSON.json(f_am_dict)

# Define the directory and file path
# output_dir = "../output/$(dataset)/transit_$(transit)"
# Define the directory and file path
output_dir = "../output/$(dataset)"

# Create the directory if it doesn't exist
if !isdir(output_dir)
    mkdir(output_dir)
end


# Optionally, save to a file
open("$output_dir/f_am.json", "w") do file
    write(file, f_am_json)
end

# If you want to print the JSON string to the console

p_ijmr_dict = Dict()
for od in ods 
    for m in m_set
        for r in keys(R[od, m])
            p_ijmr_dict[od, m, r] = p_ijmr_solution[od, m, r]
        end
    end
end

p_ijmr_json = JSON.json(p_ijmr_dict)


# Optionally, save to a file
open("$output_dir/p_ijmr.json", "w") do file
    write(file, p_ijmr_json)
end


# If you want to print the JSON string to the console

p_ijm_dict = Dict()
for od in ods 
    for m in m_set
        p_ijm_dict[od, m] = p_ijm_solution[od, m]
    end
end

p_ijm_json = JSON.json(p_ijm_dict)


# Optionally, save to a file
open("$output_dir/p_ijm.json", "w") do file
    write(file, p_ijm_json)
end