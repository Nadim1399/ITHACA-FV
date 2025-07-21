# import numpy as np
# import matplotlib.pyplot as plt

# # Carica il vettore da error.npy
# error_U = np.load('error_U.npy')
# error_p = np.load('error_p.npy')
# error_nut = np.load('error_nut.npy')

# x = np.linspace(1, 2000, len(error_U))

# # Crea il grafico
# plt.semilogy(x, error_U, label="Error_U")
# plt.semilogy(x, error_p, label="Error_p")
# plt.semilogy(x, error_nut, label="Error_nut")

# # Aggiungi etichette e titolo
# plt.xlabel("Time Step")
# plt.ylabel("Error")
# plt.title("Error between FOM and ROM")

# plt.minorticks_on()

# # Aggiungi la griglia maggiore e minore
# plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.5)

# # Puoi personalizzare la densità dei "minor ticks" per avere rettangoli più piccoli
# plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))  # Modifica la distanza tra i minor ticks sull'asse X
# plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Modifica la distanza tra i minor ticks sull'asse Y

# # Aggiungi la legenda
# plt.legend()

# # Esporta il grafico in un file png
# plt.savefig("error_plot.png")




import numpy as np
import matplotlib.pyplot as plt

# Carica gli errori
error_U = np.load('error_U.npy')
error_p = np.load('error_p.npy')
error_nut = np.load('error_nut.npy')

x = np.linspace(1, len(error_U), len(error_U)) 

def plot_error(x, error, label, filename, title):
    plt.figure() 

    plt.plot(x, error, label=label, linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Relative L2 Error")
    plt.title(title)

    plt.minorticks_on()
    plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.5)
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.legend()
    plt.tight_layout() 
    plt.savefig(filename)
    plt.close() 
    print(f"Saved: {filename}")

# Crea e salva i 3 grafici
plot_error(x, error_U, "Error_U", "error_U_plot.png", "Relative L2 Error - U")
plot_error(x, error_p, "Error_p", "error_p_plot.png", "Relative L2 Error - p")
plot_error(x, error_nut, "Error_nut", "error_nut_plot.png", "Relative L2 Error - nut")
