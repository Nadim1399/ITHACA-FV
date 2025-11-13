import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
})

# Vettore media nel tempo dell'errore per ciascuna combinazione di modi POD
# error = [0.10635220372987937, 0.03859440825257425, 0.009145598847423877, 0.003110354813706687, 0.0018633317165597304, 0.001668480026412859]
error = [0.10635220372987937, 0.03859440825257425, 0.009145598847423877, 0.003110354813706687, 0.0018633317165597304, 0.0016684800264128595, 0.0017025177859924985]   #con 20 POD

# Vettore modi POD
# POD_modes = [1, 2, 4, 6, 8, 10]
POD_modes = [1, 2, 4, 6, 8, 10, 20]    #con 20 POD

# Plot
plt.semilogy(POD_modes, error, marker='o', label="error")

# Titoli
plt.xlabel("Number of POD modes", fontsize=25)
plt.ylabel("Error", fontsize=25)
plt.title("Error with n° of POD modes variations", fontsize=25)
plt.tick_params(axis='both', labelsize=24)
plt.grid(True)
plt.tight_layout()
plt.legend(fontsize=20)
plt.savefig("error_PODmodes_plot.png")
plt.savefig("error_PODmodes_plot.pdf")