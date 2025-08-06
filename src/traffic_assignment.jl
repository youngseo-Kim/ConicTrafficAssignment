using CSV
using DataFrames
using LinearAlgebra
using PyCall
using Dates 
using Base.Threads
using Serialization
using JuMP
using Ipopt
using MosekTools
using SCS
using ECOS
using JSON

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


# Units
#The units of free flow travel time are often treated as minutes. so that given unit-less link lengths are treated as km.

dataset = length(ARGS) >= 1 ? ARGS[1] : "SiouxFalls"  # default is SiouxFalls network if not provided
solver = length(ARGS) >= 2 ? ARGS[2] : "MOSEK"        # default is MOSEK if not provided
scale_lambda = length(ARGS) >= 3 ? ARGS[3] : "1" # scaling factor   
entropy_cones = length(ARGS) >= 4 ? ARGS[4] : "exponential" # "relative" or "relative_ver2" 
beckmann = length(ARGS) >= 5 ? ARGS[5] : "power" # "power" or "socp" 
C = length(ARGS) >= 6 ? ARGS[6] : "1" # scaling factor for Beckmann SOCP   
objective = length(ARGS) >= 7 ? ARGS[7] : "original" # or "beckmann" or "entropy"  
max_n_routes = length(ARGS) >= 8 ? ARGS[8] : "3" # or "beckmann" or "entropy"  


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
println("Maximum number of routes: $max_n_routes")

network_df = CSV.read(joinpath("..", "data", dataset, dataset * "_net.txt"), DataFrame, delim='\t')
network_df = select(network_df, Not(:Column1))
network_df = dropmissing(network_df, :free_flow_time) # drop rows with missing values in the free flow time column
network_df = network_df[isfinite.(network_df.free_flow_time), :]  # keep only rows where free_flow_time is finite
od_df = CSV.read(joinpath("..", "data", dataset, dataset * "_od.csv"), DataFrame)
od_df = filter(:Ton => >(0.0), od_df) # remove all the ods pair with 0 demand 
println("Total number of active OD pairs after filtering: ", nrow(od_df))
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


println("Loading route data")
using Pandas: read_pickle
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


println("Loaded route data")



const IS_USED_FILE = "../data/$dataset/indicator.jls"

function is_continuous_subsequence(subseq::Tuple, seq::Vector)
    n = length(subseq)
    for i in 1:(length(seq) - n + 1)
        if Tuple(seq[i:i+n-1]) == subseq
            return true
        end
    end
    return false
end

routes = Dict((od, m) => keys(R[od, m]) for od in ods, m in m_set)

if isfile(IS_USED_FILE)
    println("Loading indicator function from file...")
    is_used = deserialize(IS_USED_FILE)
    println("Loaded indicator function.")
else
    println("Precomputing indicator function...")


    # Thread-safe: one dict per thread
    is_used_temp = [Dict{Tuple{Tuple{Int,Int}, Tuple{Int,Int}, String, Any}, Bool}() for _ in 1:nthreads()]

    @time begin
        @threads for ai in 1:length(A) # This part can be fasten utilizing multiple threads
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

    println("Finished precomputing indicator function. Saving to file...")
    serialize(IS_USED_FILE, is_used)
    println("Saved is_used.")
end


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

function build_model(; 
    A, ods, d_od, R, routes, 
    c_a, t0_am, alpha, beta, C, N, 
    is_used, m_set, M_set, Mm, tau_M, 
    objective::String, entropy_cones::String, beckmann::String, solver::String, scale_lambda::Float64)
    
    println(Threads.nthreads())  # This function returns the number of threads Julia is currently using

    if solver == "MOSEK"
        FS = Model(Mosek.Optimizer)
        set_optimizer_attribute(FS, "MSK_IPAR_PRESOLVE_USE", 0)
        set_optimizer_attribute(FS, "MSK_IPAR_NUM_THREADS", Threads.nthreads())
        # set_optimizer_attribute(FS, "MSK_IPAR_INTPNT_BASIS", 1)
        # set_optimizer_attribute(FS, "MSK_IPAR_INTPNT_MAX_ITERATIONS", 200)
        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-14)
        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_TOL_PFEAS", 1e-14)
        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_TOL_DFEAS", 1e-14)
        # set_optimizer_attribute(FS, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-14)
        # set_optimizer_attribute(FS, "MSK_IPAR_INTPNT_SOLVE_FORM", dual) 

        
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
        @variable(FS, 0 <= t_ij) # KL divergence should be always non-negative
        @variable(FS, 0 <= w_ijmr)
    elseif entropy_cones in ["relative", "exponential"]
        # auxiliary variable for exponential cone
        @variable(FS, 0 <= t_ij[od in ods] <= 0.3679) # maximum value of -p log p when p is in [0, 1]  
        @variable(FS, 0 <= w_ijmr[od in ods, m in m_set, r in keys(R[od, m])] <= 0.3679) #delete it later for ablation test
    end

    # Note: changed it to inequality


    if entropy_cones == "relative_ver2"
        @constraint(FS, t_ij_lb, t_ij >= -sum(d_od[i,j]/N * log(d_od[i,j]/sum(d_od[i, jp] for jp in D if (i,jp) in ods)) for (i,j) in ods if d_od[i,j] > 0)) # observed constraints
    elseif entropy_cones in ["relative", "exponential"]
        @constraint(FS, t_ij_lb, sum(t_ij[od] for od in ods) >= -sum(d_od[i,j]/N * log(d_od[i,j]/sum(d_od[i, jp] for jp in D if (i,jp) in ods)) for (i,j) in ods if d_od[i,j] > 0)) # observed constraints
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

        if objective == "original"
            @objective(FS, Max, 
            - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N * scale_lambda
            - sum(f_am_bar[a, "m1"] for a in A) / N * scale_lambda
            + t_ij
            + w_ijmr 
            # - sum(f_am_bar[a, "m1"] for a in A) / N * 1e6 # sparsity-promoting penalty
            )

        elseif objective == "entropy"
            @objective(FS, Max, 
            + t_ij
            + w_ijmr 
            )
        
        elseif objective == "beckmann"
            @objective(FS, Max, 
            - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N
            - sum(f_am_bar[a, "m1"] for a in A) / N 
            )


        end

    elseif entropy_cones in ["relative", "exponential"]

        #lambda_log = 0.001  # You may tune this penalty coefficient
        if objective == "original"

            # Full objective
            @objective(FS, Max, 
                - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta + 1)) * s_a[a] for a in A) * C / N * scale_lambda #* 1e-7 # * scale_lambda #
                - sum(f_am_bar[a, "m1"] for a in A) / N * scale_lambda #* 1e-7
                + sum(t_ij[od] for od in ods) #* 1e-5
                + sum(w_ijmr[od, m, r] for od in ods, m in m_set, r in keys(R[od, m])) #* 1e-5
                # - sum(f_am_bar[a, "m1"] for a in A) / N * 1e6 # * 1e-4 # sparsity-promoting penalty
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
            - sum(t0_am[a, "m1"] * f_am_bar[a, "m1"] + (t0_am[a, "m1"] * alpha / (c_a[a]^4) / (beta+1)) * s_a[a] for a in A) * C / N 
            - sum(f_am_bar[a, "m1"] for a in A) / N 
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
            @constraint(FS, [t_ij[od], p_ij[od], demand_sums[i]/N] in MOI.ExponentialCone()) # TODO: why negative does not come here?
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


    return FS, f_am_bar, p_ij, p_ijmr, t_ij_lb
end


println("Building model...")
build_start_time = now()

FS, f_am_bar, p_ij, p_ijmr, t_ij_lb = build_model(
        A=A, ods=ods, d_od=d_od, R=R, routes=routes,
        c_a=c_a, t0_am=t0_am, alpha=alpha, beta=beta, C=C, N=N,
        is_used=is_used, m_set=m_set, M_set=M_set, Mm=Mm, tau_M=tau_M,
        objective=objective, entropy_cones=entropy_cones, beckmann=beckmann,
        solver=solver, scale_lambda=scale_lambda
    )

build_end_time = now()
build_duration = build_end_time - build_start_time
println("Model building took $(Dates.value(build_duration)/1000) seconds")

optimize!(FS)



# Check the status of the solution
status = termination_status(FS)
println("Termination status: $status")

println("Min p_ij: ", minimum(value.(p_ij)))
println("Max p_ij: ", maximum(value.(p_ij)))

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
