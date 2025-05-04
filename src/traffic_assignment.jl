using CSV
using DataFrames
using LinearAlgebra
using PyCall

# This definition is not necessary for traffic assignment with a single transportation mode (private car), 
# but we include it for ease of future extensions.

M_set = ["M1"]  
m_set = ["m1"]  
Mm = Dict("M1" => ["m1"]) 
B_m = Dict("m1" => "M1")


theta_dest = 1
tau_M = Dict("M1" => 1)
fuel_cost = 0.5 # $/km 
value_of_time = 20 # $/hr
# fuel_cost = 0 # $/km 
# value_of_time = 1 # $/hr


# Units
#The units of free flow travel time are often treated as minutes. so that given unit-less link lengths are treated as km.

# dataset = length(ARGS) >= 1 ? ARGS[1] : "SiouxFalls" #default is SiouxFalls network if not provided
# solver = "MOSEK"
# Check if there are at least two arguments provided
dataset = length(ARGS) >= 1 ? ARGS[1] : "SiouxFalls"  # default is SiouxFalls network if not provided
solver = length(ARGS) >= 2 ? ARGS[2] : "MOSEK"        # default is MOSEK if not provided
scale_lambda = length(ARGS) >= 3 ? ARGS[3] : "1" # scaling factor   
entropy_cones = length(ARGS) >= 4 ? ARGS[4] : "exponential" # "relative" or "relative_ver2" 
beckmann = length(ARGS) >= 5 ? ARGS[5] : "power" # "power" or "socp" 
C = length(ARGS) >= 6 ? ARGS[6] : "1" # scaling factor for Beckmann SOCP   
objective = length(ARGS) >= 7 ? ARGS[7] : "original" # or "beckmann" or "entropy"  


C = parse(Int, C)
scale_lambda = parse(Float64, scale_lambda)

# Print the values to verify
println("Dataset: $dataset")
println("Solver: $solver")
println("Scale parameter Lambda: $scale_lambda")
println("Implementation for entropy functions: $entropy_cones")
println("Implementation for Beckmann: $beckmann")
println("Scale parameter for Beckmann SOCP (C): $C")
println("Objective function: $objective")

# dataset = "Munich" # "Chicago" #  #"SiouxFalls"  # "Test" #"SiouxFalls" #"Test" #"Chicago" #Note: need to change the parameter in the below

network_df = CSV.read(joinpath("..", "data", dataset, dataset * "_net.txt"), DataFrame, delim='\t')
network_df = select(network_df, Not(:Column1))
network_df = dropmissing(network_df, :free_flow_time) # drop rows with missing values in the free flow time column
network_df = network_df[isfinite.(network_df.free_flow_time), :]  # keep only rows where free_flow_time is finite
# node_df = CSV.read(joinpath("..", "data", dataset, dataset * "_node.txt"), DataFrame, delim='\t')
# node_df = select(node_df, Not(";"))
od_df = CSV.read(joinpath("..", "data", dataset, dataset * "_od.csv"), DataFrame)


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



using Pandas: read_pickle
R = read_pickle("../data/$dataset/OD_route.pickle")
route_length = read_pickle("../data/$dataset/OD_route_length.pickle")

# using PyCall
# py"""
# def is_continuous_subsequence(my_tuple, my_list):
#     tuple_length = len(my_tuple)
#     list_length = len(my_list)
    
#     for i in range(0, list_length - tuple_length + 1):
#         if tuple(my_list[i:i+tuple_length]) == my_tuple:
#             return True
#     return False

# """

# # Access the Python function as if it were a Julia function
# is_continuous_subsequence = py"is_continuous_subsequence"

using Base.Threads

function is_continuous_subsequence(subseq::Tuple, seq::Vector)
    n = length(subseq)
    for i in 1:(length(seq) - n + 1)
        if Tuple(seq[i:i+n-1]) == subseq
            return true
        end
    end
    return false
end

println("Precomputing is_used...")

routes = Dict((od, m) => keys(R[od, m]) for od in ods, m in m_set)

# Thread-safe: one dict per thread
is_used_temp = [Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}, String, Any}, Bool}() for _ in 1:nthreads()]

@time begin
    @threads for ai in 1:length(A)
        a = A[ai]
        tid = threadid()
        d = is_used_temp[tid]

        for od in ods
            for m in m_set
                for r in routes[(od, m)]
                    d[(a, od, m, r)] = is_continuous_subsequence(a, R[od, m][r])
                end
            end
        end
    end

    # Merge dictionaries
    is_used = Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}, String, Any}, Bool}()
    for d in is_used_temp
        for (k, v) in d
            is_used[k] = v
        end
    end
end

println("Finished precomputing is_used.")



# delta_a_ijm = Dict()
# for a in A
#     for od in ods
#         for m in m_set
#             delta_a_ijm[a, od, m] = 0  # Initialize here to ensure every possible key is accounted for
#             for r in routes[(od, m)]
#                 if is_continuous_subsequence(a, R[od, m][r])
#                     delta_a_ijm[a, od, m] += 1
#                 end
#             end
#         end
#     end
# end

# PS_ijmr = Dict()
# for od in ods
#     for m in m_set
#         for r in routes[(od, m)]
#             PS_ijmr[od, m, r] = 0
#             # Total length of the route
                
#             for a in A
#                 if is_continuous_subsequence(a, R[od, m][r])    
#                     PS_ijmr[od, m, r] += 1/ delta_a_ijm[a, od, m] * road_link[a]/ route_length[od, m][r]
#                 end
#             end

#         end
#     end
# end

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

using JuMP
using Ipopt
using MosekTools
using SCS
using ECOS

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
    @variable(FS, 0 <= sum_p_ijmr[od in ods, m in m_set, r in keys(R[od, m])] <= 1)
    @variable(FS, 0 <= p_ijmr[od in ods, m in m_set, r in keys(R[od, m])] <= 1)



    if entropy_cones == "relative_ver2"
        @variable(FS, 0 <= t_ij) 
        @variable(FS, 0 <= w_ijmr)
    else
        # auxiliary variable for exponential cone
        @variable(FS, 0 <= t_ij[od in ods]) 
        @variable(FS, 0 <= w_ijmr[od in ods, m in m_set, r in keys(R[od, m])])
    end

    # Note: changed it to inequality


    if entropy_cones == "relative_ver2"
        @constraint(FS, t_ij_lb, t_ij == -sum(d_od[i,j]/N * log(d_od[i,j]/sum(d_od[i, jp] for jp in D if (i,jp) in ods)) for (i,j) in ods if d_od[i,j] > 0)) # observed constraints
    else
        @constraint(FS, t_ij_lb, sum(t_ij[od] for od in ods) == -sum(d_od[i,j]/N * log(d_od[i,j]/sum(d_od[i, jp] for jp in D if (i,jp) in ods)) for (i,j) in ods if d_od[i,j] > 0)) # observed constraints
    end

    demand_sums = Dict(i => sum(d_od[i, j] for j in D if (i,j) in ods) for i in O)

  
    @constraint(FS, [i in O], sum(p_ij[(i,j)] for j in D if (i,j) in ods) == demand_sums[i]/N)
    @constraint(FS, [od in ods], p_ij[od] == sum(p_ijmr[od, m, r] for m in m_set for r in routes[(od, m)]))
    

 
    # # Objective function (PSL X)
    # @objective(FS, Max, 
    # - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N
    # - sum(t0_am[a,m]*transit_time_multiplier * f_am_bar[a, m] for m in ["m2", "m3"] for a in A) * C / N 
    # + sum(t_ij[od] for od in ods)
    # + sum(sum(v_ijm[od, m] for od in ods for m in Mm[M])*tau_M[M] for M in M_set)
    # + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m]))
    # )
    # 
    # regularization term to prevent the variable to be too small...   + 1e-6 * q_ijmr # Regularization term

    # # PSL O
    # # Objective function
    # @objective(FS, Max, 
    # + sum(V_ijmr[od, "m1", r] * p_ijmr[od, "m1", r] for od in ods, r in keys(R[od, "m1"]))/3 # $ to minutes. $20/hr = $0.33/min
    # + sum(V_ijm[od, m] * p_ijm[od, m] for od in ods, m in ["m2", "m3"])/3
    # - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N
    # - sum(t0_am[a,m]* f_am_bar[a, m] for m in ["m2", "m3"] for a in A) * C / N 
    # + sum(t_ij[od] for od in ods)
    # + sum(sum(v_ijm[od, m] for od in ods for m in Mm[M])*tau_M[M] for M in M_set)
    # + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m]))
    # )
    # # + sum(p_ijmr[od, m, r] * log(PS_ijmr[od, m, r]) for od in ods, m in m_set, r in keys(R[od, m]))

     
    # TODO: debugging PS_ijmr
    # Objective function

    # TODO: write program for dual case
    if entropy_cones == "relative_ver2"
        @objective(FS, Max, 
        - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N 
        + t_ij
        + w_ijmr 
        )

    else


        if objective == "original"
            @objective(FS, Max, 
            - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N * scale_lambda
            + sum(t_ij[od] for od in ods)
            + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m])) 
            )

        # Only with Entropy functions
        elseif objective == "entropy"
            @objective(FS, Max, 
            + sum(t_ij[od] for od in ods)
            + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m])) 
            )

        elseif objective == "beckmann"
            # # Only with Beckmann equation
            @objective(FS, Max, 
            - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N * scale_lambda
            )
        end
    end


    @constraint(FS, [a in A, m in ["m1"]], f_am_bar[a, m] * C == sum(p_ijmr[od, m, r] for od in ods for r in routes[(od, m)] if is_used[(a, od, m, r)]) * N ) #Scaling: use f_am_bar instead of f_am

   

    if entropy_cones == "relative" # RelativeEntropyCone - final
        @constraint(FS, [od in ods], [-t_ij[od], sum(d_od[first(od), j] for j in D if (first(od), j) in ods)/N, p_ij[od]] in MOI.RelativeEntropyCone(3))

        @constraint(FS, [od in ods, m in m_set, r in keys(R[od, m])], sum_p_ijmr[od, m, r] == sum(p_ijmr[od, m, rp] for rp in keys(R[od, m])))
        @constraint(FS, [od in ods, m in m_set, r in keys(R[od, m])], [-w_ijmr[od, m, r], sum_p_ijmr[od, m, r], p_ijmr[od, m, r]] in MOI.RelativeEntropyCone(3))

    elseif entropy_cones == "relative_ver2"

        @constraint(FS, vcat(-t_ij, vec([sum(d_od[first(od), j] for j in D if (first(od), j) in ods)/N for od in ods]), vec([p_ij[od] for od in ods])) in MOI.RelativeEntropyCone(2 * length(ods) +1))


        @constraint(FS, [od in ods, m in m_set, r in keys(R[od, m])], sum_p_ijmr[od, m, r] == sum(p_ijmr[od, m, rp] for rp in keys(R[od, m])))
        @constraint(FS, vcat(-w_ijmr, vec([sum_p_ijmr[od, m, r] for od in ods for m in m_set for r in routes[(od, m)]]), vec([p_ijmr[od, m, r] for od in ods for m in m_set for r in routes[(od, m)]])) in MOI.RelativeEntropyCone(2 * length([0 for od in ods for m in m_set for r in routes[(od, m)]]) + 1))




    elseif entropy_cones == "exponential" # exponential cone 
        for od in ods
            (i,_) = od
            @constraint(FS, [t_ij[od], p_ij[od], demand_sums[i]/N] in MOI.ExponentialCone())
        end


        @constraint(FS, [od in ods, m in m_set, r in keys(R[od, m])], sum_p_ijmr[od, m, r] == sum(p_ijmr[od, m, rp] for rp in keys(R[od, m])))
        @constraint(FS, [od in ods, m in m_set, r in keys(R[od, m])], [w_ijmr[od, m, r], p_ijmr[od, m, r], sum_p_ijmr[od, m, r]] in MOI.ExponentialCone())

    end
    # SOCP constraints - check if we need to change to 

    
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

f_am_solution = value.(f_am_bar * C)
p_ij_solution = value.(p_ij)
p_ijmr_solution = value.(p_ijmr)

t_ij_dual = dual.(t_ij_lb) # FirstStage


# v_ijm_dual = dual.(v_ijm_lb)

obj_value = objective_value(FS)
println("Current best solution with objective value: $obj_value")

theta_dest = 1/(t_ij_dual + 1) # FirstStage
println("theta destination: ", theta_dest) # FirstStage

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
"""



bpr_func = py"bpr_func"

using JSON

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
        for r in routes[(od, m)]
            p_ijmr_dict[od, m, r] = p_ijmr_solution[od, m, r]
        end
    end
end

p_ijmr_json = JSON.json(p_ijmr_dict)


# Optionally, save to a file
open("$output_dir/p_ijmr.json", "w") do file
    write(file, p_ijmr_json)
end
