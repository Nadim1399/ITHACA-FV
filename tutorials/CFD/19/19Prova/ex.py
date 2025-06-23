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




from smithers.io.openfoam import OpenFoamHandler
import numpy as np
import matplotlib.pyplot as plt

FOM_data = OpenFoamHandler().read("./ITHACAoutput/Offline/", time_instants="all_numeric")
ROM_data = OpenFoamHandler().read("./ITHACAoutput/POD/", time_instants="all_numeric")

# Lista dei time step come stringhe
time_steps_ROM = [str(i) for i in range(1, 11)]
time_steps_FOM = [str(i) for i in range(1, 2001)]

N_expected = 125000

# Funzione per estrarre dati puliti (u, p, nut)
def extract_clean_data(data, time_steps):
    u_list, p_list, nut_list = [], [], []
    for t in time_steps:
        fields = data[t]["fields"]

        u_flat = np.array(fields["U"][1]).flatten()
        p_flat = np.array(fields["p"][1]).flatten()
        nut_flat = np.array(fields["nut"][1]).flatten()

        # Controllo shape su tutti e tre prima di aggiungere
        if u_flat.shape == (N_expected,) and p_flat.shape == (N_expected,) and nut_flat.shape == (N_expected,):
            u_list.append(u_flat)
            p_list.append(p_flat)
            nut_list.append(nut_flat)
        else:
            print(f"❌ Skip time {t}: shapes u={u_flat.shape}, p={p_flat.shape}, nut={nut_flat.shape}")

    return np.array(u_list), np.array(p_list), np.array(nut_list)

# Estrai dati puliti FOM e ROM
u_matrix_FOM, p_matrix_FOM, nut_matrix_FOM = extract_clean_data(FOM_data, time_steps_FOM)
u_matrix_ROM, p_matrix_ROM, nut_matrix_ROM = extract_clean_data(ROM_data, time_steps_ROM)

# Allineamento temporale FOM->ROM per confronto
step_ratio = len(u_matrix_FOM) // len(u_matrix_ROM)
u_matrix_FOM = u_matrix_FOM[::step_ratio]
p_matrix_FOM = p_matrix_FOM[::step_ratio]
nut_matrix_FOM = nut_matrix_FOM[::step_ratio]

time_FOM = np.arange(len(u_matrix_FOM))
time_ROM = np.arange(len(u_matrix_ROM))

print(f"Shapes dopo pulizia e sottocampionamento:")
print(f"u_matrix_FOM: {u_matrix_FOM.shape}")
print(f"u_matrix_ROM: {u_matrix_ROM.shape}")

# Punto da plottare (ad esempio punto centrale)
idx = N_expected // 2

# Plot U
plt.figure(figsize=(12,6))
plt.plot(time_FOM, u_matrix_FOM[:, idx], label='FOM - U', color='blue')
plt.plot(time_ROM, u_matrix_ROM[:, idx], label='ROM - U', color='orange')
plt.title('FOM vs ROM - Velocità U')
plt.xlabel('Time step')
plt.ylabel(f'Valore in indice = {idx}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_U_FOM_vs_ROM.png', dpi=300)
plt.show()

# Plot p
plt.figure(figsize=(12,6))
plt.plot(time_FOM, p_matrix_FOM[:, idx], label='FOM - p', color='blue')
plt.plot(time_ROM, p_matrix_ROM[:, idx], label='ROM - p', color='orange')
plt.title('FOM vs ROM - Pressione p')
plt.xlabel('Time step')
plt.ylabel(f'Valore in indice = {idx}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_p_FOM_vs_ROM.png', dpi=300)
plt.show()

# Plot nut
plt.figure(figsize=(12,6))
plt.plot(time_FOM, nut_matrix_FOM[:, idx], label='FOM - nut', color='blue')
plt.plot(time_ROM, nut_matrix_ROM[:, idx], label='ROM - nut', color='orange')
plt.title('FOM vs ROM - Coefficiente nut')
plt.xlabel('Time step')
plt.ylabel(f'Valore in indice = {idx}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_nut_FOM_vs_ROM.png', dpi=300)
plt.show()

# Calcolo errore relativo medio (norma L2) per U
rel_error_u = np.linalg.norm(u_matrix_FOM - u_matrix_ROM, axis=1) / np.linalg.norm(u_matrix_FOM, axis=1)
plt.figure(figsize=(10,5))
plt.plot(time_FOM, rel_error_u, label='Errore relativo medio su U')
plt.xlabel('Time step')
plt.ylabel('Errore relativo')
plt.title('Errore relativo medio tra FOM e ROM per U')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('error_U_FOM_vs_ROM.png', dpi=300)
plt.show()

# Matrici di correlazione temporale (per verifica)
coeff_matrix_u = np.corrcoef(u_matrix_FOM, u_matrix_ROM)[:len(u_matrix_FOM), len(u_matrix_FOM):]
coeff_matrix_p = np.corrcoef(p_matrix_FOM, p_matrix_ROM)[:len(p_matrix_FOM), len(p_matrix_FOM):]
coeff_matrix_nut = np.corrcoef(nut_matrix_FOM, nut_matrix_ROM)[:len(nut_matrix_FOM), len(nut_matrix_FOM):]

np.save("coeff_matrix_u.npy", coeff_matrix_u)
np.save("coeff_matrix_p.npy", coeff_matrix_p)
np.save("coeff_matrix_nut.npy", coeff_matrix_nut)
