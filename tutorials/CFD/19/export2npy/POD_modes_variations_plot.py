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

# Caricare i campi FOM #
uFOM_POD_matrix = np.load('./FOM_POD/U_FOM_POD.npy')
pFOM_POD_matrix = np.load('./FOM_POD/P_FOM_POD.npy')
nutFOM_POD_matrix = np.load('./FOM_POD/Nut_FOM_POD.npy')

print ("uFOM_POD_matrix_shape", uFOM_POD_matrix.shape)
print ("pFOM_POD_matrix_shape", pFOM_POD_matrix.shape)
print ("nutFOM_POD_matrix_shape", nutFOM_POD_matrix.shape)

# Caricare i campi ed i coefficienti ROM #
uROM_field_LSTM = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/u_field_LSTM.npy')
CoeffU_mat = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/Coeffs_rom/CoeffU_mat.npy')


pROM_field_LSTM = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/p_field_LSTM.npy')
CoeffP_mat = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/Coeffs_rom/CoeffP_mat.npy')

nutROM_field_LSTM = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/nut_field_LSTM.npy')
CoeffNut_mat = np.load('/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/Coeffs_rom/CoeffNut_mat.npy')


# Calcola pseudoinversa dei coefficienti temporali
CoeffU_inv = np.linalg.pinv(CoeffU_mat)
CoeffP_inv = np.linalg.pinv(CoeffP_mat)
CoeffNut_inv = np.linalg.pinv(CoeffNut_mat)

# Ricava la matrice dei modi POD
uROM_POD_matrix = uROM_field_LSTM @ CoeffU_inv
pROM_POD_matrix = pROM_field_LSTM @ CoeffP_inv
nutROM_POD_matrix = nutROM_field_LSTM @ CoeffNut_inv

print ("uROM_POD_matrix_shape", uROM_POD_matrix.shape)
print ("pROM_POD_matrix_shape", pROM_POD_matrix.shape)
print ("nutROM_POD_matrix_shape", nutROM_POD_matrix.shape)

np.save("ROM_POD/uROM_POD_matrix.npy", uROM_POD_matrix)
np.save("ROM_POD/pROM_POD_matrix.npy", pROM_POD_matrix)
np.save("ROM_POD/nutROM_POD_matrix.npy", nutROM_POD_matrix)

# Calcolo dell'erorre #
errors = np.sum(np.abs(pFOM_POD_matrix - pROM_POD_matrix), axis=0) / np.sum(np.abs(pFOM_POD_matrix), axis=0)
print("Errori per ciascun modo POD:")
print(errors)

#  Plot #

plt.figure(figsize=(7,4))
plt.plot(np.arange(1, 11), errors, 'o-', color='teal')
plt.xlabel('Modo POD')
plt.ylabel('Errore relativo')
plt.title('Errore relativo |FOM - ROM| per ciascun modo POD')
plt.grid(True)
plt.show()
plt.savefig("plot.png")