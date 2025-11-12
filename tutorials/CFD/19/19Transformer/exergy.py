import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
})

# Carica dati FOM e ROM
fom_u = np.load("./Coeffs/u_coeffs.npy")[:, 1:]  
fom_p = np.load("./Coeffs/p_coeffs.npy")[:, 1:]  

rom_u = np.load("./Coeffs_rom/CoeffU_mat.npy")
rom_p = np.load("./Coeffs_rom/CoeffP_mat.npy")

# Parametri 
rho = 1.225 

def calc_exergy_proxy(coeff_u, coeff_p, rho=1.225):
    kin_energy = 0.5 * rho * np.sum(coeff_u**2, axis=0)  
    press_energy = np.sum(coeff_p**2, axis=0)
    return kin_energy + press_energy  

# Calcola estrofia (proxy) per FOM e ROM
fom_psi = calc_exergy_proxy(fom_u, fom_p, rho)
rom_psi = calc_exergy_proxy(rom_u, rom_p, rho)

# Calcolo errore
abs_error = np.abs(fom_psi - rom_psi)
rel_error = abs_error / (np.abs(fom_psi) + 1e-8)

# Plot errore relativo nel tempo
# plt.figure(figsize=(10, 4))
plt.plot(rel_error, marker='o', markevery=200, label='Relative exstrophy error')
plt.xlabel("Snapshot", fontsize=25)
plt.ylabel("Relative error ", fontsize=25)
plt.grid(True)
plt.legend(fontsize=20)
plt.title("Transformer exstrophy error", fontsize=25)
plt.tick_params(axis='both', labelsize=24)
plt.tight_layout()
plt.show()
plt.savefig("./Exergy/Exergy_Tr.png")
plt.savefig("./Exergy/Exergy_Tr.pdf")

# # Histogramma degli errori
# plt.hist(rel_error, bins=40, alpha=0.7, label='Relative Error')
# plt.xlabel("Errore relativo estrofia")
# plt.ylabel("Frequenza")
# plt.legend()
# plt.title("Distribuzione dell'errore relativo tra FOM e ROM")
# plt.grid(True)
# plt.show()
# plt.savefig("Exergy_hystogram.png")

# Statistiche
print("Errore medio assoluto:", np.mean(abs_error))
print("Errore relativo medio:", np.mean(rel_error))
# errore relativo medio = 0.018053281774792495