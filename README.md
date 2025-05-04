# Traffic Assignment via Conic Reformulation

This repository provides a tutorial implementation of conic reformulations for the traffic assignment problem. It explores how the **Beckmann** formulation and **entropy-based** user equilibrium can be expressed using **exponential** and **second-order cones**, solved with conic solvers such as MOSEK, ECOS, and SCS.

---

## 🚀 How to Run the Code

Run the following commands in your terminal using [Julia](https://julialang.org/):

```bash
julia traffic_assignment.jl SiouxFalls MOSEK 1 exponential power 1 original
julia traffic_assignment.jl Munich MOSEK 1 exponential power 1 original
julia traffic_assignment.jl Chicago MOSEK 1 exponential power 1 original
```

### Arguments

1. **Network**: `SiouxFalls`, `Anaheim`, `EMA`, `Munich`, `Chicago`  
2. **Solver**: `MOSEK`, `ECOS`, `SCS`  
3. **Scale Parameter for Lambda**: Scales the magnitude of the entropy function and Beckmann equation  
4. **Reformulation for Entropy Function**: `exponential`, `relative`, `relative_ver2`  
5. **Reformulation for Beckmann Equation**: `power`, `socp`  
6. **Scale Parameter for Beckmann SOCP**: A constant ≥ 1  
7. **Objective Function**: `original`, `beckmann`, `entropy`

---

## 🔍 Best Configuration

Based on a comprehensive evaluation, the best-performing configuration uses:
- **Exponential cone** for entropy functions  
- **Power cone** for the Beckmann equation  
- **MOSEK** as the solver

To determine an appropriate **lambda scale parameter**, solve the problem using only the Beckmann equation and only the entropy function, respectively. Then, tune lambda to ensure both components operate on a similar scale.

---

## 📂 Data

The datasets are downloaded from the [TransportationNetworks repository](https://github.com/bstabler/TransportationNetworks) and slightly modified for consistency in data types.

---

## 👤 Author

**Youngseo Kim**  
✉️ [youngseo@ucla.edu](mailto:youngseo@ucla.edu)
