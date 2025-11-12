# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Computer Modern",
# })

# U_FOM_LSTM = np.load('U.npy')
# U_ROM_LSTM = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/u_field_LSTM.npy')

# P_FOM_LSTM = np.load('P.npy')
# P_ROM_LSTM = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/p_field_LSTM.npy')

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
})

# --- Carica i campi ---
P_FOM_LSTM = np.load('P.npy')           # (125000, 2001)
P_ROM_LSTM = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/p_field_LSTM.npy')  # (125000, 2000)

# --- Allinea nel tempo (salta il primo snapshot del FOM) ---
P_FOM_LSTM = P_FOM_LSTM[:, 1:]   # ora (125000, 2000)

print("Shape P_FOM:", P_FOM_LSTM.shape)
print("Shape P_ROM:", P_ROM_LSTM.shape)
print("Valori P_FOM:", np.min(P_FOM_LSTM), np.max(P_FOM_LSTM))
print("Valori P_ROM:", np.min(P_ROM_LSTM), np.max(P_ROM_LSTM))
print("Somma assoluta P_FOM:", np.sum(np.abs(P_FOM_LSTM)))

# --- Calcolo errore integrale normalizzato ---
num = np.sum(np.abs(P_FOM_LSTM - P_ROM_LSTM))   # integrale doppio della differenza
den = np.sum(np.abs(P_FOM_LSTM))                # integrale doppio del riferimento
err = num / den

print(f"Errore integrale normalizzato: {err:.6e}")

# --- (opzionale) Grafico dell'errore per snapshot ---
err_t = np.sum(np.abs(P_FOM_LSTM - P_ROM_LSTM), axis=0) / np.sum(np.abs(P_FOM_LSTM), axis=0)

plt.figure(figsize=(6,4))
plt.plot(err, color='navy', lw=1.5)
# plt.plot(np.arange(1, P_FOM_LSTM.shape[1] + 1), err, color='navy', lw=1.5)
plt.xlabel('Snapshot (tempo)')
plt.ylabel('Errore normalizzato')
plt.title('Errore di pressione nel tempo')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig('plot')



