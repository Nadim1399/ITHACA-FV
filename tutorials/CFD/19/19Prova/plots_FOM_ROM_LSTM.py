import numpy as np
import matplotlib.pyplot as plt

###### PLOT FOM_ROM_LSTM ######

coeffs_u = np.load("./Coeffs/u_coeffs.npy")
coeffs_p = np.load("./Coeffs/p_coeffs.npy")
coeffs_nut = np.load("./Coeffs/nut_coeffs.npy")

t = np.arange(0, 2001).reshape(1,-1)

### Errori ###
error_U_FOM_ROM = (coeffs_u -)

# Plot errore U
plt.figure(figsize=(12, 6))
for idx in range(matrix.shape[0]):
    plt.plot(t.flatten(), error_U_FOM_ROM[idx], label=f'errore FOM-ROM {idx}')
    plt.plot(t.flatten(), error_U_FOM_LSTM[idx], label=f'errore FOM-(ROM+LSTM) {idx}')
plt.title('Error U')
plt.xlabel('Time step')
plt.ylabel(f'Valori {name}')
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_error_U.png', dpi=300)
plt.show()

# Plot errore p
plt.figure(figsize=(12, 6))
for idx in range(matrix.shape[0]):
    plt.plot(t.flatten(), error_p_FOM_ROM[idx], label=f'errore FOM-ROM {idx}')
    plt.plot(t.flatten(), error_p_FOM_LSTM[idx], label=f'errore FOM-(ROM+LSTM) {idx}')
plt.title('Error p')
plt.xlabel('Time step')
plt.ylabel(f'Valori {name}')
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_error_p.png', dpi=300)
plt.show()

# Plot errore nut
plt.figure(figsize=(12, 6))
for idx in range(matrix.shape[0]):
    plt.plot(t.flatten(), error_nut_FOM_ROM[idx], label=f'errore FOM-ROM {idx}')
    plt.plot(t.flatten(), error_nut_FOM_LSTM[idx], label=f'errore FOM-(ROM+LSTM) {idx}')
plt.title('Error nut')
plt.xlabel('Time step')
plt.ylabel(f'Valori {name}')
plt.legend(loc='upper right', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_error_nut.png', dpi=300)
plt.show()



