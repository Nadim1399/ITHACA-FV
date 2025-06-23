import numpy as np
from scipy.linalg import solve
from ReducedUnsteadyNSExplicit import ReducedUnsteadyNSExplicit


# Definire il metodo di flusso da usare: "consistent" o "inconsistent"
flux_method = "consistent"

# Velocità di input (può essere float o array, a seconda del modello)
vel = 1.0

# Istanziare il solver
solver = ReducedUnsteadyNSExplicit(flux_method)

# Esecuzione della simulazione
solver.solve_online(vel)

# Filtro per rimuovere i None da online_solution
# cleaned_solution = [arr for arr in solver.online_solution if arr is not None]

# Verifica che tutte le righe abbiano la stessa shape
# shapes = [arr.shape for arr in cleaned_solution]
shapes = [arr.shape for arr in solver.online_solution if arr is not None]

for i, arr in enumerate(solver.online_solution):
    print(f"Elemento {i}: shape = {np.shape(arr)}")

# Se tutte le righe hanno la stessa shape, si salva come array 2D
if all(s == shapes[0] for s in shapes):
    # online_solution_array = np.array(cleaned_solution)
    # np.save("online_solution_" + str(flux_method) + ".npy", online_solution_array)
    online_solution_array = np.array([arr for arr in solver.online_solution if arr is not None])
    np.save("online_solution_" + str(flux_method) + ".npy", online_solution_array)
else:
    # Altrimenti si salva come oggetto (dtype=object)
    # np.save("online_solution_" + str(flux_method) + ".npy", np.array(cleaned_solution, dtype=object))
    np.save("online_solution_" + str(flux_method) + ".npy", np.array([arr for arr in solver.online_solution if arr is not None], dtype=object))








# import numpy as np
# import os
# from ReducedUnsteadyNSExplicit import ReducedUnsteadyNSExplicit

# # === Parametri ===
# flux_method = "consistent"
# vel = 1.0
# save_folder = "coeffs_output"
# os.makedirs(save_folder, exist_ok=True)

# # === solver e simulazione ===
# solver = ReducedUnsteadyNSExplicit(flux_method)
# solver.solve_online(vel)

# # Debug
# print(f"Lunghezza online_solution: {len(solver.online_solution)}")
# print(f"Lunghezza nut_coeffs_history: {len(solver.nut_coeffs_history)}")

# # Debug
# print(f"Nphi_u = {solver.Nphi_u}, Nphi_p = {solver.Nphi_p}")

# # === Inizializza array coefficienti ===
# CoeffU = []
# CoeffP = []
# CoeffNut = []
# tValues = []

# for i, (sol, nut_coeff) in enumerate(zip(solver.online_solution, solver.nut_coeffs_history)):
#     if sol is None or nut_coeff is None:
#         print(f"Salto step {i}: sol=None o nut_coeff=None")
#         continue

#     try:
#         time_now = sol[0]
#         Nphi_u = solver.Nphi_u[0] if isinstance(solver.Nphi_u, (list, np.ndarray)) else solver.Nphi_u
#         Nphi_p = solver.Nphi_p[0] if isinstance(solver.Nphi_p, (list, np.ndarray)) else solver.Nphi_p

#         currentUCoeff = sol[1 : 1 + Nphi_u].reshape(-1, 1)
#         currentPCoeff = sol[1 + Nphi_u : 1 + Nphi_u + Nphi_p].reshape(-1, 1)
#         currentNutCoeff = nut_coeff.reshape(-1, 1)

#         CoeffU.append(currentUCoeff)
#         CoeffP.append(currentPCoeff)
#         CoeffNut.append(currentNutCoeff)
#         tValues.append(time_now)
#     except Exception as e:
#         print(f" Errore al time step {i}: {e}")

# # === Salva se non sono vuoti ===
# if CoeffU:
#     np.save(os.path.join(save_folder, "CoeffU.npy"), np.array(CoeffU))
#     np.save(os.path.join(save_folder, "CoeffP.npy"), np.array(CoeffP))
#     np.save(os.path.join(save_folder, "CoeffNut.npy"), np.array(CoeffNut))
#     np.save(os.path.join(save_folder, "tValues.npy"), np.array(tValues))
#     print(f" Coefficienti salvati in '{save_folder}/'")
# else:
#     print(" Nessun coefficiente salvato. Verifica se la simulazione è andata a buon fine.")

