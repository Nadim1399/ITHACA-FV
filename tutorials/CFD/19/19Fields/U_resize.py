import numpy as np

u_resize = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/u_field_LSTM.npy")

u_reshaped = u_resize.reshape(-1, 2000)

np.save("./u_reshaped.npy", u_reshaped)
print(u_reshaped.shape)
# print(u_resize.shape)
print(u_resize)
print("stop")
print(u_reshaped)

