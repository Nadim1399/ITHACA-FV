# from smithers.io.openfoam import OpenFoamHandler
# import numpy as np
# import matplotlib.pyplot as plt

# FOM_data = OpenFoamHandler().read("./ITHACAoutput/Offline/", time_instants="all_numeric")
# ROM_data = OpenFoamHandler().read("./ITHACAoutput/POD/", time_instants="all_numeric")

# # Lista dei time step come stringhe
# time_steps_ROM = [str(i) for i in range(1, 11)]
# time_steps_FOM = [str(i) for i in range(1, 2001)]

# # Inizializzare le liste di matrici per ogni variabile 
# u_matrix_FOM = []
# p_matrix_FOM = []
# # phi_matrix_FOM = []
# nut_matrix_FOM = []

# u_matrix_ROM = []
# p_matrix_ROM = []
# # phi_matrix_ROM = []
# nut_matrix_ROM = []

# for t in time_steps_FOM:
#     fields = FOM_data[t]["fields"]

#     u_data = np.array(fields["U"][1]) 
#     u_flat = u_data.flatten() 
#     u_matrix_FOM.append(u_flat)    

#     p_data = np.array(fields["p"][1]) 
#     p_flat = p_data.flatten() 
#     p_matrix_FOM.append(p_flat)

#     nut_data = np.array(fields["nut"][1]) 
#     nut_flat = nut_data.flatten()
#     if nut_flat.shape != (125000,):
#         print(f"❌ Skip time {t}: shape nut = {nut_flat.shape}")
#         continue  # salta il timestep
#     nut_matrix_FOM.append(nut_flat)


# for t in time_steps_ROM:
#     fields = ROM_data[t]["fields"]

#     u_data = np.array(fields["U"][1]) 
#     u_flat = u_data.flatten() 
#     u_matrix_ROM.append(u_flat)    

#     p_data = np.array(fields["p"][1]) 
#     p_flat = p_data.flatten() 
#     p_matrix_ROM.append(p_flat)

#     nut_data = np.array(fields["nut"][1]) 
#     nut_flat = nut_data.flatten()
#     nut_matrix_ROM.append(nut_flat)

# N_expected = 125000

# # FOM #
# u_matrix_FOM_clean = []
# p_matrix_FOM_clean = []
# nut_matrix_FOM_clean = []

# for u, p, nut in zip(u_matrix_FOM, p_matrix_FOM, nut_matrix_FOM):
#     if u.shape[0] == N_expected and p.shape[0] == N_expected and nut.shape[0] == N_expected:
#         u_matrix_FOM_clean.append(u)
#         p_matrix_FOM_clean.append(p)
#         nut_matrix_FOM_clean.append(nut)

# u_matrix_FOM = np.array(u_matrix_FOM_clean)
# p_matrix_FOM = np.array(p_matrix_FOM_clean)
# nut_matrix_FOM = np.array(nut_matrix_FOM_clean)

# # ROM #
# u_matrix_ROM_clean = []
# p_matrix_ROM_clean = []
# nut_matrix_ROM_clean = []

# for u, p, nut in zip(u_matrix_ROM, p_matrix_ROM, nut_matrix_ROM):
#     if u.shape[0] == N_expected and p.shape[0] == N_expected and nut.shape[0] == N_expected:
#         u_matrix_ROM_clean.append(u)
#         p_matrix_ROM_clean.append(p)
#         nut_matrix_ROM_clean.append(nut)

# # Conversione in array NumPy
# u_matrix_ROM = np.array(u_matrix_ROM_clean)
# p_matrix_ROM = np.array(p_matrix_ROM_clean)
# nut_matrix_ROM = np.array(nut_matrix_ROM_clean)

# print("u_matrix_FOM shape:", u_matrix_FOM.shape)
# print("u_matrix_ROM shape:", u_matrix_ROM.shape)
# print("Tipo degli elementi:", type(u_matrix_FOM[0]))

# time_FOM = np.arange(u_matrix_FOM.shape[0])
# time_ROM = np.arange(u_matrix_ROM.shape[0])

# idx = 0

# # --- Plot U ---
# plt.figure(figsize=(12, 6))
# plt.plot(time_FOM, u_matrix_FOM[:, idx], label='FOM - U', color='blue')
# plt.plot(time_ROM, u_matrix_ROM[:, idx], label='ROM - U', color='orange')
# plt.title('FOM vs ROM - U')
# plt.xlabel('Time step')
# plt.ylabel('Valore in idx = 0')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('plot_U_FOM_vs_ROM.png', dpi=300)
# plt.show()

# # --- Plot p ---
# plt.figure(figsize=(12, 6))
# plt.plot(time_FOM, p_matrix_FOM[:, idx], label='FOM - p', color='blue')
# plt.plot(time_ROM, p_matrix_ROM[:, idx], label='ROM - p', color='orange')
# plt.title('FOM vs ROM - p')
# plt.xlabel('Time step')
# plt.ylabel('Valore in idx = 0')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('plot_p_FOM_vs_ROM.png', dpi=300)
# plt.show()

# # --- Plot nut ---
# plt.figure(figsize=(12, 6))
# plt.plot(time_FOM[:len(nut_matrix_FOM)], nut_matrix_FOM[:, idx], label='FOM - nut', color='blue')
# plt.plot(time_ROM, nut_matrix_ROM[:, idx], label='ROM - nut', color='orange')
# plt.title('FOM vs ROM - nut')
# plt.xlabel('Time step')
# plt.ylabel('Valore in idx = 0')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('plot_nut_FOM_vs_ROM.png', dpi=300)
# plt.show()

# # Matrici dei coefficienti
# coeff_matrix_u = u_matrix_FOM @ u_matrix_ROM.T
# coeff_matrix_p = p_matrix_FOM @ p_matrix_ROM.T
# # coeff_matrix_phi = phi_matrix_FOM @ phi_matrix_ROM
# coeff_matrix_nut = nut_matrix_FOM @ nut_matrix_ROM.T

# # Save delle matrici in formato numpy 
# np.save("coeff_matrix_u" + ".npy", coeff_matrix_u)
# np.save("coeff_matrix_p" + ".npy", coeff_matrix_p)
# # np.save("./Matrix_py/coeff_matrix_phi" + ".npy", coeff_matrix_phi)
# np.save("coeff_matrix_nut" + ".npy", coeff_matrix_nut)


################################################################################################################################


# from smithers.io.openfoam import OpenFoamHandler
# import numpy as np
# import matplotlib.pyplot as plt
# import os, psutil

# # Parametri
# N_expected = 125000
# idx = N_expected // 2  # Punto centrale

# # Riduci il numero di time step FOM (ad esempio 1 ogni 20)
# time_steps_ROM = [str(i) for i in range(1, 11)]
# time_steps_FOM = [str(i) for i in range(1, 2001, 200)]  # max 100 step

# print(f"🔍 Time steps FOM selezionati: {len(time_steps_FOM)}")

# # Funzione ottimizzata: estrai solo il punto idx
# def extract_data_at_point(data, time_steps, idx):
#     u_vals, p_vals, nut_vals = [], [], []
#     for t in time_steps:
#         fields = data[t]["fields"]

#         if "U" not in fields:
#             print(f"⚠️ Skip time {t}: campo 'U' mancante")
#             continue

#         u_raw = np.array(fields["U"][1])

#         if u_raw.size != N_expected * 3:
#             print(f"⚠️ Skip time {t}: dimensione U = {u_raw.size}, attesa = {N_expected * 3}")
#             continue

#         try:
#             u_vec = u_raw.reshape((N_expected, 3))
#         except ValueError:
#             print(f"❌ Impossibile fare reshape per il time {t}, shape U: {u_raw.shape}")
#             continue

#         u_norm = np.linalg.norm(u_vec, axis=1)

#         p_flat = np.array(fields["p"][1]).flatten()
#         nut_flat = np.array(fields["nut"][1]).flatten()

#         if (
#             u_norm.shape == (N_expected,)
#             and p_flat.shape == (N_expected,)
#             and nut_flat.shape == (N_expected,)
#         ):
#             u_vals.append(u_norm[idx])
#             p_vals.append(p_flat[idx])
#             nut_vals.append(nut_flat[idx])
#         else:
#             print(f"❌ Skip time {t} per shape inconsistente")

#     return np.array(u_vals), np.array(p_vals), np.array(nut_vals)

# # Leggi solo una volta per evitare duplicati
# FOM_data = OpenFoamHandler().read("./ITHACAoutput/Offline/", time_instants=time_steps_FOM)
# ROM_data = OpenFoamHandler().read("./ITHACAoutput/POD/", time_instants=time_steps_ROM)

# # Estrai solo i valori nel punto `idx`
# u_FOM, p_FOM, nut_FOM = extract_data_at_point(FOM_data, time_steps_FOM, idx)
# u_ROM, p_ROM, nut_ROM = extract_data_at_point(ROM_data, time_steps_ROM, idx)

# # Associa il tempo
# time_FOM = np.arange(len(u_FOM))
# time_ROM = np.arange(len(u_ROM))

# # Plot U
# plt.figure(figsize=(12, 6))
# plt.semilogy(time_FOM, u_FOM, label="FOM - U", color="blue")
# plt.semilogy(time_ROM, u_ROM, label="ROM - U", color="orange")
# plt.title("FOM vs ROM - Velocità U (norma) in idx")
# plt.xlabel("Time step")
# plt.ylabel("Valore")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plot_U_FOM_vs_ROM.png", dpi=300)
# plt.show()

# # Plot p
# plt.figure(figsize=(12, 6))
# plt.semilogy(time_FOM, p_FOM, label="FOM - p", color="blue")
# plt.semilogy(time_ROM, p_ROM, label="ROM - p", color="orange")
# plt.title("FOM vs ROM - Pressione p in idx")
# plt.xlabel("Time step")
# plt.ylabel("Valore")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plot_p_FOM_vs_ROM.png", dpi=300)
# plt.show()

# # Plot nut
# plt.figure(figsize=(12, 6))
# plt.semilogy(time_FOM, nut_FOM, label="FOM - nut", color="blue")
# plt.semilogy(time_ROM, nut_ROM, label="ROM - nut", color="orange")
# plt.title("FOM vs ROM - nut in idx")
# plt.xlabel("Time step")
# plt.ylabel("Valore")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plot_nut_FOM_vs_ROM.png", dpi=300)
# plt.show()

# # Calcolo errore relativo su U (solo nel punto)
# min_len = min(len(u_FOM), len(u_ROM))
# rel_error_u = np.abs(u_FOM[:min_len] - u_ROM[:min_len]) / (np.abs(u_FOM[:min_len]) + 1e-10)

# plt.figure(figsize=(10, 5))
# plt.semilogy(np.arange(min_len), rel_error_u, label="Errore relativo medio su U")
# plt.xlabel("Time step")
# plt.ylabel("Errore relativo")
# plt.title("Errore relativo medio tra FOM e ROM (punto idx)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("error_U_FOM_vs_ROM.png", dpi=300)
# plt.show()


################################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from smithers.io.openfoam import OpenFoamHandler

# --- Parametri ---
N_cells = 125000
N_components = 3
idx = N_cells // 2
grid_shape = (250, 500)  # per le heatmap

# Time steps
time_steps_FOM = [str(i) for i in range(1, 2001)]
time_steps_ROM = [str(i) for i in range(1, 11)]

# Carica dati
FOM_data = OpenFoamHandler().read("./ITHACAoutput/Offline/", time_instants=time_steps_FOM)
ROM_data = OpenFoamHandler().read("./ITHACAoutput/POD/", time_instants=time_steps_ROM)


# --- Funzione di estrazione ---
def extract_fields(data, time_steps, N_expected=125000):
    U_list, p_list, nut_list = [], [], []
    for t in time_steps:
        fields = data[t]["fields"]

        u_raw = np.array(fields["U"][1])
        if u_raw.ndim == 1 and u_raw.size == N_expected * 3:
            u_vec = u_raw.reshape((N_expected, 3))
        elif u_raw.ndim == 2 and u_raw.shape == (N_expected, 3):
            u_vec = u_raw
        else:
            print(f"❌ Errore su U al tempo {t}: shape {u_raw.shape}")
            continue

        u_norm = np.linalg.norm(u_vec, axis=1)
        p_flat = np.array(fields["p"][1]).flatten()
        nut_flat = np.array(fields["nut"][1]).flatten()

        if (
            u_norm.shape == (N_expected,)
            and p_flat.shape == (N_expected,)
            and nut_flat.shape == (N_expected,)
        ):
            U_list.append(u_norm)
            p_list.append(p_flat)
            nut_list.append(nut_flat)
        else:
            print(f"❌ Skip time {t} per shape mismatch")
    return np.array(U_list), np.array(p_list), np.array(nut_list)


# --- Estrai dati ---
u_FOM, p_FOM, nut_FOM = extract_fields(FOM_data, time_steps_ROM)  
u_ROM, p_ROM, nut_ROM = extract_fields(ROM_data, time_steps_ROM)

# --- Funzione plotting ---
def plot_comparison(FOM, ROM, var_name):
    # Errore L2 nel tempo
    error_L2 = np.linalg.norm(FOM - ROM, axis=1) / np.linalg.norm(FOM, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(FOM) + 1), error_L2, marker='o')
    plt.title(f"Errore relativo L2 tra FOM e ROM - {var_name}")
    plt.xlabel("Snapshot #")
    plt.ylabel("Errore relativo L2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"error_L2_{var_name}.png", dpi=300)
    plt.show()

    # Serie temporale al punto centrale
    FOM_point = FOM[:, idx]
    ROM_point = ROM[:, idx]

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(FOM) + 1), FOM_point, label=f"FOM - {var_name}", color='blue')
    plt.plot(range(1, len(ROM) + 1), ROM_point, label=f"ROM - {var_name}", color='orange')
    plt.title(f"FOM vs ROM - {var_name} al punto centrale")
    plt.xlabel("Snapshot #")
    plt.ylabel(var_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"timeseries_{var_name}.png", dpi=300)
    plt.show()

    # Heatmap dell’errore spaziale (snapshot finale)
    diff_snapshot = np.abs(FOM[-1] - ROM[-1])
    diff_image = diff_snapshot[:grid_shape[0] * grid_shape[1]].reshape(grid_shape)

    plt.figure(figsize=(6, 5))
    plt.imshow(diff_image, cmap='hot')
    plt.colorbar(label=f"|FOM - ROM| {var_name}")
    plt.title(f"Errore spaziale {var_name} (snapshot finale)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"heatmap_{var_name}.png", dpi=300)
    plt.show()


# --- Esegui confronto per U, p, nut ---
plot_comparison(u_FOM, u_ROM, "U")
plot_comparison(p_FOM, p_ROM, "p")
plot_comparison(nut_FOM, nut_ROM, "nut")
