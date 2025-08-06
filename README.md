# Convex Programming Approach for Travel Demand Modeling

This repository provides a tutorial implementation of (1) a combined convex model for destination, mode, and route choice, and (2) a convex traffic assignment problem. It explores how the **Beckmann** formulation and **entropy-based** user equilibrium can be expressed through conic formulations, such as **exponential** and **second-order cones**, and solved using conic solvers like MOSEK, ECOS, and SCS.

---

## 📦 Installation

To set up the environment:

1. Clone the repository and navigate into the folder.
2. Create the Conda environment using:

```bash
conda env create -f environment.yml
conda activate traffic_assignment
```

3. You will need to obtain an **academic license** for [MOSEK](https://www.mosek.com/products/academic-licenses/) and install it according to their instructions. To use the academic license, you may need to use your institution's VPN.


## Project Structure

1. `data/`: We utilize the transportation benchmark networks obtained from [this source](https://github.com/bstabler/TransportationNetworks). The inputs are standardized, with `NETWORKNAME_net.txt` representing the network links and `NETWORKNAME_od.csv` representing the OD matrix processed from `NETWORKNAME_trips.tntp`. Networks include Sioux Falls, Eastern Massachusetts (EMA), Berlin Friedrichshain, Berlin Mitte, Anaheim, Barcelona, Winnipeg, and Chicago Sketch. To download the full data directory with preprocessed files, download and unzip the following directory, and replace the existing `data` directory. [data]https://drive.google.com/file/d/1lavAvhQqSgyFXaFmYS8CVicf8OU5znF7/view?usp=sharing

2. `src/`: Main implementation to build and solve the model. Follow the guidelines for more details. 
    - `candidate_set.py`: Creates the candidate route set and saves the output as `OD_route_ROUTENUMBER.pickle` and `OD_route_length_ROUTENUMBER.pickle`.
    - `preprocessing.ipynb`: Converts the `NETWORKNAME_trips.tntp` file to `NETWORKNAME_od.csv`.
    - `first_stage_scaled.jl`: Solves the combined demand model including destination, mode, and route choices.
    - `traffic_assignment.jl`: Solves the traffic assignment problem. 
    - `visualize_solution.ipynb`: Visualizes the output of the model. Reads the optimal solution from the `output/` directory. 
- `log/`: Stores logging files. 
- `models/`: Stores optimization models as MPS files. These can be reused later to save model-building time.  
- `output/`: Stores output solutions after solving the optimization problem. 


---

## 🚀 How to Run the Code

1. Precompute candidate sets. Go to the directory `src/` and run

```bash
python candidate_set.py --data_type SiouxFalls --n_routes 3
python candidate_set.py --data_type Anaheim --n_routes 3
python candidate_set.py --data_type EMA --n_routes 3
```

2. Run the following commands in your terminal using [Julia](https://julialang.org/):

### Commands for traffic assignment: 

```bash
julia traffic_assignment.jl SiouxFalls MOSEK 1 exponential power 1 original 3  > ../log/SiouxFalls_route3
julia traffic_assignment.jl Anaheim MOSEK 1 exponential power 1 original 3 > ../log/Anaheim_route3
julia traffic_assignment.jl EMA MOSEK 1 exponential power 1 original 3 > ../log/EMA_route3
```


### Arguments for traffic assignment: 

1. **Network**: `SiouxFalls`, `Anaheim`, `EMA`, `Munich`, `Chicago`, `Barcelona`, `BerlinFriedrichshain`, `BerlinMitte`, `Winnipeg`
2. **Solver**: `MOSEK`, `ECOS`, `SCS`  
3. **Scale Parameter for Lambda**: Scales the magnitude of the entropy function and Beckmann equation  
4. **Reformulation for Entropy Function**: `exponential`, `relative`, `relative_ver2`  
5. **Reformulation for Beckmann Equation**: `power`, `socp`  
6. **Scale Parameter for Beckmann SOCP**: A constant ≥ 1  
7. **Objective Function**: `original`, `beckmann`, `entropy`
8. **Number of Candidate Routes**: Typical range would be 2-10


### Commands for the combined model:

```bash
JULIA_NUM_THREADS=4 julia first_stage_scaled.jl SiouxFalls MOSEK first false false 1 1000 exponential power original 3 > ../log/TDM/SiouxFalls
JULIA_NUM_THREADS=4 julia first_stage_scaled.jl Anaheim MOSEK first false false 1 0.5 exponential power original 3 > ../log/TDM/Anaheim
JULIA_NUM_THREADS=4 julia first_stage_scaled.jl EMA MOSEK first false false 1 7000 exponential power original 3 > ../log/TDM/EMA 
```


### Arguments for the combined model:
1. **Network**: `SiouxFalls`, `Anaheim`, `EMA`, `Munich`, `Chicago`, `Barcelona`, `BerlinFriedrichshain`, `BerlinMitte`, `Winnipeg`
2. **Solver**: `MOSEK`, `ECOS`, `SCS`  
3. **Stage**: `first`, `second` 
4. **Whether data is perturb**: `true`, `false` 
5. **Whether transit network is provided**: `true`, `false` 
6. **Scale Parameter for Beckmann SOCP**: A constant ≥ 1  
7. **Scale Parameter for Lambda**: Scales the magnitude of the entropy function and Beckmann equation  
8. **Reformulation for Entropy Function**: `exponential`, `relative`, `relative_ver2`  
9. **Reformulation for Beckmann Equation**: `power`, `socp`  
10. **Objective Function**: `original`, `beckmann`, `entropy`
11. **Number of Candidate Routes**: Typical range would be 2-10



---

## 🔍 Best Configuration

Based on a comprehensive evaluation, the best-performing configuration uses:
- **Exponential cone** for entropy functions  
- **Power cone** for the Beckmann equation  
- **MOSEK** as the solver

To determine an appropriate **lambda scale parameter**, solve the problem using only the Beckmann equation and only the entropy function, respectively. Then, tune lambda to ensure both components operate on a similar scale.

---


## 👤 Author

**Youngseo Kim**  
✉️ [youngseo@ucla.edu](mailto:youngseo@ucla.edu)
