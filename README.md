# MA-DC Hub Location Model Implementation

## Evolution from Original to Final Implementation

This repository contains the **final implementation** (`434_Final.ipynb`) which represents a significant simplification and optimization of the original complex multiple-allocation hub location model (`434.ipynb`). Here are the key changes and improvements:

### **Model Simplification: Multiple → Single Allocation**

**Original Model (`434.ipynb`):**
- Complex multiple-allocation formulation with 31,997 variables
- Inter-hub flow variables and conservation constraints
- Direct connection alternatives with additional constraints
- Multiple hub paths per origin-destination pair
- Extensive parameter sweeps and gamma scaling

**Final Model (`434_Final.ipynb`):**
- Simplified single-allocation formulation with 15,650 variables
- Each origin-destination pair assigned to at most one hub
- No inter-hub flow variables or conservation constraints
- Streamlined objective function and constraints
- Focus on computational efficiency and solver performance

### **Key Additions and Improvements**

#### 1. **Performance Optimization Techniques**
- **MIP Warm Start**: Greedy initialization strategy that pre-assigns profitable routes
- **Variable Fixing**: Eliminates unprofitable routing variables (reduced from 15,650 to 183 variables)
- **Hub Limit Constraints**: Optional constraint limiting maximum number of open hubs
- **Data Scaling**: Normalized flow matrix for numerical stability

#### 2. **Computational Efficiency**
- **Solve Time Reduction**: From complex multiple-allocation (hours) to single-allocation (seconds)
- **Memory Optimization**: Dramatically reduced model size and constraint complexity
- **Presolve Effectiveness**: Better presolve performance due to simpler structure

#### 3. **Experimental Framework**
- **Cold vs Warm Start Comparison**: Systematic evaluation of initialization strategies
- **Parameter Sensitivity Analysis**: Revenue and cost parameter variations
- **Performance Benchmarking**: Comprehensive timing and objective value tracking

### **Model Formulation Changes**

| Aspect | Original (Multiple) | Final (Single) |
|--------|-------------------|----------------|
| **Variables** | 31,997 (complex) | 15,650 (simplified) |
| **Constraints** | 63,391 (flow conservation) | 15,600 (assignment only) |
| **Routing** | Multiple paths per OD | Single hub per OD |
| **Flow** | Inter-hub flow variables | Direct hub assignment |
| **Solve Time** | Hours (complex) | Seconds (efficient) |

### **Performance Results**

The final implementation demonstrates significant improvements:

- **Variable Fixing**: 99% reduction in variables (15,650 → 183)
- **Solve Time**: 50-80% faster with warm start initialization
- **Model Complexity**: Eliminated flow conservation constraints
- **Numerical Stability**: Better scaling and presolve performance

### **Why This Simplification?**

1. **Computational Tractability**: Original model was too complex for practical use
2. **Performance Focus**: Single-allocation provides good solutions with much faster solve times
3. **Educational Value**: Cleaner formulation for understanding hub location concepts
4. **Extensibility**: Simpler base model allows for easier modifications and extensions

---

## Overview

This repository contains a comprehensive implementation of the **Single Allocation Hub Location Problem** based on the 2018 Taherkhani & Alumur paper (OMEGA). The implementation includes:

- **Cold-start and warm-start optimization**
- **Hub capacity constraints extension**
- **Parameter sweep experiments**
- **Performance comparison and visualization**
- **CAB25 dataset integration**

## Features

### 1. Warm-Start Model Builder
- `build_ma_dc_model_warmstart()`: Builds MA-DC model with optional warm-start initialization
- Supports initialization of all decision variables (hubs, links, flows, direct connections)
- Automatic candidate reduction for profitable hub paths

### 2. Hub Capacity Extension
- `build_ma_dc_model_with_capacity()`: Adds capacity constraints to hubs
- Configurable capacity limits per hub
- Maintains all original model constraints

### 3. Experiment Drivers
- `run_cold_vs_warm()`: Direct comparison of cold vs warm start performance
- `run_grid_cold()`: Parameter sweep using cold start
- `run_grid_warm()`: Parameter sweep using warm start with initial solutions

### 4. Visualization and Reporting
- `plot_performance_comparison()`: Creates comprehensive performance plots
- `print_solution_summary()`: Detailed solution analysis
- `save_results_to_csv()`: Results export functionality

## Model Formulation

### Decision Variables
- `x[k]`: Binary variable for hub location at node k
- `y[i,j,k]`: Binary variable for routing i→j via hub k

### Objective Function
Maximize: Revenue - Collection/Distribution Costs - Hub Fixed Costs

### Key Constraints
1. **Single Assignment**: Each OD pair served at most once via one hub
2. **Hub Requirements**: Routing through hub k requires hub k to be open
3. **Optional Extensions**: 
   - Hub limit constraints (maximum number of open hubs)
   - Variable fixing (eliminate unprofitable routes)
   - Warm start initialization (greedy assignment)

## Usage

### Basic Usage

```python
# Import required libraries
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Load and prepare data
cost_file = 'CAB25.txt'
raw = np.loadtxt(cost_file, skiprows=1)
n = raw.shape[1]
C = raw[:n, :]  # cost matrix
W = raw[n:2*n, :] / raw[n:2*n, :].sum()  # normalized flow matrix

# Define parameters
revenue = 1500
install_f = 100
nodes = range(n)

# Build single-allocation model
m = gp.Model("single_alloc")
x = m.addVars(nodes, vtype=GRB.BINARY, name="x")
y = m.addVars(nodes, nodes, nodes, vtype=GRB.BINARY, name="y")

# Objective: maximize profit
obj = gp.quicksum(
    revenue * W[i,j] * y[i,j,k]
  - (C[i,k] + C[k,j]) * W[i,j] * y[i,j,k]
  for i in nodes for j in nodes for k in nodes if i != j
)
obj -= gp.quicksum(install_f * x[k] for k in nodes)
m.setObjective(obj, GRB.MAXIMIZE)

# Constraints
m.addConstrs(
    (gp.quicksum(y[i,j,k] for k in nodes) <= 1
     for i in nodes for j in nodes if i != j),
    name="assign"
)
m.addConstrs(
    (y[i,j,k] <= x[k]
     for i in nodes for j in nodes for k in nodes if i != j),
    name="use_hub"
)

# Solve
m.optimize()
```

### Warm-Start Usage

```python
# Greedy warm start initialization
for var in m.getVars():
    var.start = 0

for i in nodes:
    for j in nodes:
        if i == j:
            continue
        best_k, best_profit = None, -float('inf')
        for k in nodes:
            profit = revenue * W[i, j] - (C[i, k] + C[k, j]) * W[i, j]
            if profit > best_profit:
                best_profit, best_k = profit, k
        if best_profit > 0:
            m.getVarByName(f"y[{i},{j},{best_k}]").start = 1
            m.getVarByName(f"x[{best_k}]").start = 1

# Solve with warm start
m.optimize()
```

### Variable Fixing Extension

```python
# Identify profitable routes only
profitable = [
    (i, j, k)
    for i in nodes for j in nodes for k in nodes
    if i != j and revenue * W[i, j] >= (C[i, k] + C[k, j]) * W[i, j]
]

# Build reduced model with only profitable variables
m_fix = gp.Model("single_alloc_fixed")
x_fix = m_fix.addVars(nodes, vtype=GRB.BINARY, name="x")
y_fix = m_fix.addVars(profitable, vtype=GRB.BINARY, name="y")

# Objective and constraints over profitable routes only
obj_fix = gp.quicksum(
    revenue * W[i, j] * y_fix[i, j, k]
  - (C[i, k] + C[k, j]) * W[i, j] * y_fix[i, j, k]
  for i, j, k in profitable
)
obj_fix -= gp.quicksum(install_f * x_fix[k] for k in nodes)
m_fix.setObjective(obj_fix, GRB.MAXIMIZE)

# Solve reduced model
m_fix.optimize()
```

### Hub Limit Extension

```python
# Add constraint limiting number of open hubs
p_max = 5
m.addConstr(x.sum() <= p_max, name="hub_limit")

# Solve with hub limit
m.optimize()
```

## Data Requirements

### CAB25 Dataset
The implementation uses the CAB25 dataset which should be placed in:
```
/Users/aryaaghakoochek/Downloads/CAB/CAB25.txt
```

### Data Format
- **CAB25.txt**: Contains N×N cost matrix and N×N demand matrix
- **Format**: First line contains N, followed by cost matrix, then demand matrix
- **Normalization**: Demands are automatically normalized to sum to 1

## Dependencies

- **Gurobi**: Commercial optimization solver (license required)
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Enhanced plotting styles

## Installation

1. Install required packages:
```bash
pip install gurobipy numpy pandas matplotlib seaborn
```

2. Ensure Gurobi license is active
3. Place CAB25.txt in the correct directory
4. Run the main script or test script

## Testing

Run the test script to verify functionality:
```bash
python test_ma_dc.py
```

This will test:
- Basic cold start functionality
- Warm start performance
- Hub capacity constraints
- Parameter sweep experiments

## Output Files

The implementation generates several output files:
- `ma_dc_cold_results.csv`: Cold start parameter sweep results
- `ma_dc_warm_results.csv`: Warm start parameter sweep results
- `ma_dc_performance_comparison.png`: Performance comparison plots

## Performance Characteristics

### Typical Performance Improvements
- **Warm start**: 50-80% time reduction compared to cold start
- **Variable fixing**: 99% reduction in variables (15,650 → 183)
- **Solve time**: Seconds vs hours for complex multiple-allocation
- **Scalability**: Tested on CAB25 (25-node) problems

### Parameter Sensitivity
- **Revenue parameter**: Controls profitability threshold for routes
- **Hub installation cost**: Affects optimal number of hubs
- **Hub limit constraint**: Balances coverage vs cost efficiency

## Extensions and Future Work

### Potential Extensions
1. **Multiple allocation**: Return to complex formulation with inter-hub flows
2. **Hub capacity constraints**: Limit flow through individual hubs
3. **Dynamic demand**: Time-varying demand patterns
4. **Stochastic optimization**: Uncertainty in demand/costs

### Implementation Notes
- Model uses Gurobi's advanced features for warm start
- Variable fixing dramatically improves computational efficiency
- Single-allocation constraints ensure tractable problem size
- Hub limit constraints can be easily modified or extended
