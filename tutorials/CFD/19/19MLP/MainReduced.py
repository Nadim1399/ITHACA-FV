import os
import numpy as np
from scipy.linalg import solve
from ReducedUnsteadyNSExplicit import ReducedUnsteadyNSExplicit

MLP_Rec= "MLP_Rec"
os.makedirs(MLP_Rec, exist_ok=True)

# Definire il metodo di flusso da usare: "consistent" o "inconsistent"
flux_method = "consistent"

# Velocità di input (può essere float o array, a seconda del modello)
vel = 1.0

# Istanziare il solver
solver = ReducedUnsteadyNSExplicit(flux_method)

# Esecuzione della simulazione
solver.solve_online(vel)

solver.reconstruct(MLP_Rec)

# Verifica che tutte le righe abbiano la stessa shape
shapes = [arr.shape for arr in solver.online_solution if arr is not None]

for i, arr in enumerate(solver.online_solution):
    print(f"Elemento {i}: shape = {np.shape(arr)}")

# Se tutte le righe hanno la stessa shape, si salva come array 2D
online_solution_clean = [arr for arr in solver.online_solution if arr is not None]
if all(s == shapes[0] for s in shapes):
    online_solution_array = np.array(online_solution_clean)
    np.save(os.path.join(MLP_Rec, f"online_solution_{flux_method}.npy"), online_solution_array)
else:
    np.save(os.path.join(MLP_Rec, f"online_solution_{flux_method}.npy"), 
            np.array(online_solution_clean, dtype=object))
