import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
})

# Carica coefficienti POD 
fom_u = np.load("./Coeffs/u_coeffs.npy")[:, 1:] 
rom_u = np.load("./Coeffs_rom/CoeffU_mat.npy")  

rho = 1.225 

# Calcolo energia cinetica 
def compute_kinetic_energy(coeff_u, rho=1.225):
    return 0.5 * rho * np.sum(coeff_u**2, axis=0)

# Energia per FOM e ROM
energy_fom = compute_kinetic_energy(fom_u, rho)
energy_rom = compute_kinetic_energy(rom_u, rho)

# Errore relativo
rel_error_energy = np.abs(energy_fom - energy_rom) / (np.abs(energy_fom) + 1e-8)

# #Plot energia
# plt.plot(energy_fom, label="FOM")
# plt.plot(energy_rom, label="ROM", linestyle='--')
# plt.xlabel("Snapshot")
# plt.ylabel("Energia cinetica")
# plt.title("Energia cinetica FOM vs ROM")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig("Energy_t.png")

# Plot errore relativo
plt.plot(rel_error_energy, marker='o', markevery=200, label="Relative energy error")
plt.xlabel("Snapshot", fontsize=25)
plt.ylabel("Relative error", fontsize=25)
plt.title("MLP Relative energy error", fontsize=25)
plt.tick_params(axis='both', labelsize=24)
plt.tight_layout()
plt.grid(True)
plt.legend(fontsize=20)
plt.show()
# plt.savefig("./Energy/Error_Energy_MLP.png")
plt.savefig("./Energy/Error_Energy_MLP.pdf")

# Statistiche
print("Errore medio relativo sull'energia:", np.mean(rel_error_energy))
# errore medio relativo sull'energia = 0.008316053196854292
