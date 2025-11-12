# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Computer Modern",
# })

# # Caricamento dati
# fom_coeffs_u = np.load("Coeffs/u_coeffs.npy")      # shape: (nModes, nSnapshots)
# rom_coeffs_u = np.load("Coeffs_rom/CoeffU_mat.npy")

# fom_coeffs_p = np.load("Coeffs/p_coeffs.npy")
# rom_coeffs_p = np.load("Coeffs_rom/CoeffP_mat.npy")

# fom_coeffs_nut = np.load("Coeffs/nut_coeffs.npy")
# rom_coeffs_nut = np.load("Coeffs_rom/CoeffNut_mat.npy")

# # Trasponi se necessario: vogliamo (nSnapshots, nModes)
# if fom_coeffs_u.shape[0] < fom_coeffs_u.shape[1]:
#     fom_coeffs_u = fom_coeffs_u.T
#     rom_coeffs_u = rom_coeffs_u.T
#     fom_coeffs_p = fom_coeffs_p.T
#     rom_coeffs_p = rom_coeffs_p.T
#     fom_coeffs_nut = fom_coeffs_nut.T
#     rom_coeffs_nut = rom_coeffs_nut.T

# fom_coeffs_u = fom_coeffs_u[1:, :]
# fom_coeffs_p = fom_coeffs_p[1:, :]
# fom_coeffs_nut = fom_coeffs_nut[1:, :]

# nModes_U = fom_coeffs_u.shape[1]
# nModes_p = fom_coeffs_p.shape[1]
# nModes_nut = fom_coeffs_nut.shape[1]

# # Inizializza vettori per gli errori
# errorModes_U = np.zeros(nModes_U)
# errorModes_p = np.zeros(nModes_p)
# errorModes_nut = np.zeros(nModes_nut)

# # --- Velocità U ---
# for i in range(nModes_U):
#     num = np.linalg.norm(fom_coeffs_u[:, i] - rom_coeffs_u[:, i])
#     den = np.linalg.norm(fom_coeffs_u[:, i])
#     errorModes_U[i] = num / den if den != 0 else np.nan

# # --- Pressione p ---
# for i in range(nModes_p):
#     num = np.linalg.norm(fom_coeffs_p[:, i] - rom_coeffs_p[:, i])
#     den = np.linalg.norm(fom_coeffs_p[:, i])
#     errorModes_p[i] = num / den if den != 0 else np.nan

# # --- Viscosità turbolenta nut ---
# for i in range(nModes_nut):
#     num = np.linalg.norm(fom_coeffs_nut[:, i] - rom_coeffs_nut[:, i])
#     den = np.linalg.norm(fom_coeffs_nut[:, i])
#     errorModes_nut[i] = num / den if den != 0 else np.nan

# # --- Plot confronto u ---
# plt.figure(figsize=(10, 4))
# plt.semilogy(range(1, nModes_U+1), errorModes_U, label='velocity', marker='o')
# plt.title("$L_2$ relative error per mode (MLP)", fontsize=25)
# plt.xlabel("POD modes", fontsize=25)
# plt.ylabel("Relative $L_2$ error", fontsize=25)
# plt.tick_params(axis='both', labelsize=20)
# plt.grid(True)
# plt.tight_layout()
# plt.legend(fontsize=20, loc='lower right')
# plt.savefig("confronto_u_MLP.png", dpi=300)
# plt.savefig("confronto_u_MLP.pdf")
# plt.show()

# # --- Plot confronto p ---
# plt.figure(figsize=(10, 4))
# plt.semilogy(range(1, nModes_p+1), errorModes_p, label='pressure', marker='s')
# plt.title("$L_2$ relative error per mode (MLP)", fontsize=25)
# plt.xlabel("POD modes", fontsize=25)
# plt.ylabel("Relative $L_2$ error", fontsize=25)
# plt.tick_params(axis='both', labelsize=20)
# plt.grid(True)
# plt.tight_layout()
# plt.legend(fontsize=20, loc='lower right')
# plt.savefig("confronto_p_MLP.png", dpi=300)
# plt.savefig("confronto_p_MLP.pdf")
# plt.show()

# # --- Plot confronto nut ---
# plt.figure(figsize=(10, 4))
# plt.semilogy(range(1, nModes_nut+1), errorModes_nut, label='turbulent viscosity', marker='^')
# plt.title("$L_2$ relative error per mode (MLP)", fontsize=25)
# plt.xlabel("POD modes", fontsize=25)
# plt.ylabel("Relative $L_2$ error", fontsize=25)
# plt.tick_params(axis='both', labelsize=20)
# plt.grid(True)
# plt.tight_layout()
# plt.legend(fontsize=20, loc='lower right')
# plt.savefig("confronto_nut_MLP.png", dpi=300)
# plt.savefig("confronto_nut_MLP.pdf")
# plt.show()





















import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
})

# Caricamento dati
fom_coeffs_u = np.load("Coeffs/u_coeffs.npy")  
rom_coeffs_u = np.load("Coeffs_rom/CoeffU_mat.npy")        

fom_coeffs_p = np.load("Coeffs/p_coeffs.npy")  
rom_coeffs_p = np.load("Coeffs_rom/CoeffP_mat.npy")  

fom_coeffs_nut = np.load("Coeffs/nut_coeffs.npy")  
rom_coeffs_nut = np.load("Coeffs_rom/CoeffNut_mat.npy")  

# Estrazione prima riga
row_fom_u = fom_coeffs_u[:, 2000]   
row_rom_u = rom_coeffs_u[:, 1999]   

row_fom_p = fom_coeffs_p[:, 2000]   
row_rom_p = rom_coeffs_p[:, 1999]  

row_fom_nut = fom_coeffs_nut[:, 2000]   
row_rom_nut = rom_coeffs_nut[:, 1999]  

x = np.arange(1, 11)

# Plot confronto u
plt.figure(figsize=(10, 4))
plt.plot(x, row_fom_u, marker='o', label="FOM u")
plt.plot(x, row_rom_u, marker='s', label="ROM u", linestyle='--')
plt.legend(fontsize=20)
plt.title("Coeff. u (FOM vs ROM)", fontsize=25)
plt.xlabel("POD modes", fontsize=25)
plt.ylabel("Coeffs", fontsize=25)
plt.tick_params(axis='both', labelsize=24)
plt.grid(True)
plt.tight_layout()
plt.savefig("confronto_u_MLP.png")
plt.savefig("confronto_u_MLP.pdf")
plt.close()

# Plot confronto p
plt.figure(figsize=(10, 4))
plt.plot(x, row_fom_p, marker='o', label="FOM p")
plt.plot(x, row_rom_p, marker='s', label="ROM p", linestyle='--')
plt.legend(fontsize=20)
plt.title("Coeff. p (FOM vs ROM)",fontsize=25)
plt.xlabel("POD modes",fontsize=25)
plt.ylabel("Coeffs",fontsize=25)
plt.tick_params(axis='both', labelsize=24)
plt.grid(True)
plt.tight_layout()
plt.savefig("confronto_p_MLP.png")
plt.savefig("confronto_p_MLP.pdf")
plt.close()

# Plot confronto nut
plt.figure(figsize=(10, 4))
plt.plot(x, row_fom_nut, marker='o', label="FOM nut")
plt.plot(x, row_rom_nut, marker='s', label="ROM nut", linestyle='--')
plt.legend(fontsize=20)
plt.title("Coeff. nut (FOM vs ROM)", fontsize=25)
plt.xlabel("POD modes", fontsize=25)
plt.ylabel("Coeffs", fontsize=25)
plt.tick_params(axis='both', labelsize=24)
plt.grid(True)
plt.tight_layout()
plt.savefig("confronto_nut_MLP.png")
plt.savefig("confronto_nut_MLP.pdf")
plt.close()

### Calcolo dell'errore sul primo modo ###
rel_error_u = np.linalg.norm(row_fom_u - row_rom_u) / np.linalg.norm(row_fom_u)
rel_error_p = np.linalg.norm(row_fom_p - row_rom_p) / np.linalg.norm(row_fom_p)
rel_error_nut = np.linalg.norm(row_fom_nut - row_rom_nut) / np.linalg.norm(row_fom_nut)

print(f"Errore relativo coeff. u   : {rel_error_u:.3e}")
print(f"Errore relativo coeff. p   : {rel_error_p:.3e}")
print(f"Errore relativo coeff. nut : {rel_error_nut:.3e}")

relative_errors = np.array([rel_error_u, rel_error_p, rel_error_nut])

np.save("relative_errors_MLP.npy", relative_errors)