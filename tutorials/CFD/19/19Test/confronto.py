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
row_fom_u = fom_coeffs_u[5, 1:]   
row_rom_u = rom_coeffs_u[5, :]   

row_fom_p = fom_coeffs_p[5, 1:]   
row_rom_p = rom_coeffs_p[5, :]  

row_fom_nut = fom_coeffs_nut[5, 1:]   
row_rom_nut = rom_coeffs_nut[5, :]  

# Plot confronto u
plt.figure(figsize=(10, 4))
plt.semilogy(row_fom_u, label="FOM u")
plt.semilogy(row_rom_u, label="ROM u", linestyle='--')
plt.legend()
plt.title("Confronto coeff. u (FOM vs ROM)")
plt.xlabel("Indice")
plt.ylabel("Valore (scala log)")
plt.grid(True)
plt.savefig("confronto_u.png")
plt.close()

# Plot confronto p
plt.figure(figsize=(10, 4))
plt.semilogy(row_fom_p, label="FOM p")
plt.semilogy(row_rom_p, label="ROM p", linestyle='--')
plt.legend()
plt.title("Confronto coeff. p (FOM vs ROM)")
plt.xlabel("Indice")
plt.ylabel("Valore (scala log)")
plt.grid(True)
plt.savefig("confronto_p.png")
plt.close()

# Plot confronto nut
plt.figure(figsize=(10, 4))
plt.semilogy(row_fom_nut, label="FOM nut")
plt.semilogy(row_rom_nut, label="ROM nut", linestyle='--')
plt.legend()
plt.title("Confronto coeff. nut (FOM vs ROM)")
plt.xlabel("Indice")
plt.ylabel("Valore (scala log)")
plt.grid(True)
plt.savefig("confronto_nut.png")
plt.close()

### Calcolo dell'errore sul primo modo ###
rel_error_u = np.linalg.norm(row_fom_u - row_rom_u) / np.linalg.norm(row_fom_u)
rel_error_p = np.linalg.norm(row_fom_p - row_rom_p) / np.linalg.norm(row_fom_p)
rel_error_nut = np.linalg.norm(row_fom_nut - row_rom_nut) / np.linalg.norm(row_fom_nut)

print(f"Errore relativo coeff. u   : {rel_error_u:.3e}")
print(f"Errore relativo coeff. p   : {rel_error_p:.3e}")
print(f"Errore relativo coeff. nut : {rel_error_nut:.3e}")

relative_errors = np.array([rel_error_u, rel_error_p, rel_error_nut])

np.save("relative_errors.npy", relative_errors)