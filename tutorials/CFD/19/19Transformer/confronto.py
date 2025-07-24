import numpy as np
import matplotlib.pyplot as plt

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

# Plot confronto u
plt.figure(figsize=(10, 4))
plt.plot(row_fom_u, label="FOM u")
plt.plot(row_rom_u, label="ROM u", linestyle='--')
plt.legend()
plt.title("Coeff. u (FOM vs ROM)")
plt.xlabel("POD modes")
plt.ylabel("Error")
plt.grid(True)
plt.savefig("confronto_u_Tr.png")
plt.close()

# Plot confronto p
plt.figure(figsize=(10, 4))
plt.plot(row_fom_p, label="FOM p")
plt.plot(row_rom_p, label="ROM p", linestyle='--')
plt.legend()
plt.title("Coeff. p (FOM vs ROM)")
plt.xlabel("POD modes")
plt.ylabel("Error")
plt.grid(True)
plt.savefig("confronto_p_Tr.png")
plt.close()

# Plot confronto nut
plt.figure(figsize=(10, 4))
plt.plot(row_fom_nut, label="FOM nut")
plt.plot(row_rom_nut, label="ROM nut", linestyle='--')
plt.legend()
plt.title("Coeff. nut (FOM vs ROM)")
plt.xlabel("POD modes")
plt.ylabel("Error")
plt.grid(True)
plt.savefig("confronto_nut_Tr.png")
plt.close()

### Calcolo dell'errore sul primo modo ###
rel_error_u = np.linalg.norm(row_fom_u - row_rom_u) / np.linalg.norm(row_fom_u)
rel_error_p = np.linalg.norm(row_fom_p - row_rom_p) / np.linalg.norm(row_fom_p)
rel_error_nut = np.linalg.norm(row_fom_nut - row_rom_nut) / np.linalg.norm(row_fom_nut)

print(f"Errore relativo coeff. u   : {rel_error_u:.3e}")
print(f"Errore relativo coeff. p   : {rel_error_p:.3e}")
print(f"Errore relativo coeff. nut : {rel_error_nut:.3e}")

relative_errors = np.array([rel_error_u, rel_error_p, rel_error_nut])

np.save("relative_errors_Tr.npy", relative_errors)