import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.losses import Huber
from tensorflow.keras import regularizers
import joblib
import os

# === CREA CARTELLE RISULTATI ===
os.makedirs("MLP/plots", exist_ok=True)

# ==== PARAMETRI ====
epochs = 1200
batch_size = 64   # fino a 64

# ==== CARICAMENTO DATI POD ====
coeffs_u = np.load("./Coeffs/u_coeffs.npy")     # shape (T, r_u)
coeffs_p = np.load("./Coeffs/p_coeffs.npy")     # shape (T, r_p)
coeffs_nut = np.load("./Coeffs/nut_coeffs.npy") # shape (T, r_nut)

eigen_u = np.loadtxt("./ITHACAoutput/POD/Eigenvalues_U", skiprows=2)
eigen_p = np.loadtxt("./ITHACAoutput/POD/Eigenvalues_p", skiprows=2)
eigen_nut = np.loadtxt("./ITHACAoutput/POD/Eigenvalues_nut", skiprows=2)

# ==== CONSIDERARE SOLO E PRIME 1800 COLONNE (SNAPSHOTS) PER IL TRAINING ====
u_coeffs_1800 = coeffs_u[:, :1800]
p_coeffs_1800 = coeffs_p[:, :1800]
nut_coeffs_1800 = coeffs_nut[:, :1800]

# ==== CONSIDERARE LE ULTIME 201 COLONNE (SNAPSHOTS) PER LA VALIDATION ====
u_coeffs_last201 = coeffs_u[:, -201:]
p_coeffs_last201 = coeffs_p[:, -201:]
nut_coeffs_last201 = coeffs_nut[:, -201:]

# ==== CONCATENAZIONE INPUT ====
X_all = np.hstack([u_coeffs_1800.T, p_coeffs_1800.T])  
Y_all = nut_coeffs_1800.T  

# ==== NORMALIZZAZIONE PER TRAINING E VALIDAZIONE ====
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_all = x_scaler.fit_transform(X_all)                         # Training
Y_all = y_scaler.fit_transform(Y_all)                         # Training

X_val_raw = np.hstack([u_coeffs_last201.T, p_coeffs_last201.T])   # Validazione
Y_val_raw = nut_coeffs_last201.T                                # Validazione

X_val = x_scaler.transform(X_val_raw)                         # Validazione
Y_val = y_scaler.transform(Y_val_raw)                         # Validazione

joblib.dump(x_scaler, "./MLP/x_scaler.pkl")
joblib.dump(y_scaler, "./MLP/y_scaler.pkl")

# === DEFINIZIONE MODELLO MLP ===
model_mlp = Sequential()
model_mlp.add(Dense(128, activation='relu', input_shape=(X_all.shape[1],)))
model_mlp.add(Dropout(0.2))
model_mlp.add(Dense(64, activation='relu'))
model_mlp.add(Dense(Y_all.shape[1])) 

model_mlp.compile(optimizer=Adam(learning_rate=3e-5), loss='mse')

# === TRAINING ===
history_mlp = model_mlp.fit(X_all, Y_all, validation_data=(X_val, Y_val), epochs=1000, 
                            batch_size=64, verbose=1)

# === PREDIZIONE ===
Y_train_pred_mlp = model_mlp.predict(X_all)
Y_val_pred_mlp = model_mlp.predict(X_val)

# === INVERSE TRANSFORM ===
Y_train_pred_orig = y_scaler.inverse_transform(Y_train_pred_mlp)
Y_train_true_orig = y_scaler.inverse_transform(Y_all)

Y_val_pred_orig = y_scaler.inverse_transform(Y_val_pred_mlp)
Y_val_true_orig = y_scaler.inverse_transform(Y_val)

# === SALVATAGGI ===
model_mlp.save("MLP/mlp_model.keras")
np.save("MLP/Y_val_pred_mlp.npy", Y_val_pred_orig)
np.save("MLP/Y_val_true_mlp.npy", Y_val_true_orig)
np.save("MLP/training_history_mlp.npy", history_mlp.history)

### PLOT ###

plt.figure(figsize=(8, 4))
plt.plot(history_mlp.history['loss'], label='Train Loss')
plt.plot(history_mlp.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training History - MLP")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("MLP/plots/training_history_mlp.png")
plt.close()

# ==== PLOT ERRORE Assoluto TRA VALORE PREDETTO E QUELLO ORIGINALE DOPO IL TRAINING ====
for i in range(min(3, Y_train_true_orig.shape[1])):  
    abs_error_train = np.abs(Y_train_true_orig[:, i] - Y_train_pred_orig[:, i])

    plt.figure(figsize=(10, 4))
    plt.plot(abs_error_train, label=f'|Errore training|')
    plt.title(f'Errore Assoluto (Training)')
    plt.xlabel("Time step")
    plt.ylabel("Errore assoluto")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"MLP/plots/abs_error_train.png")
    plt.show()
    plt.close()

# === PLOT ERRORE Relativo DOPO LA VALIDAZIONE ===
for i in range(Y_val_true_orig.shape[1]):
    rel_error_val = np.abs(Y_val_true_orig[:, i] - Y_val_pred_orig[:, i])/np.abs(Y_val_pred_orig[:, i])

    plt.figure(figsize=(10, 4))
    plt.plot(rel_error_val, label='|True - Pred|')
    plt.title(f'Errore Relativo (Validation)')
    plt.xlabel("Time step")
    plt.ylabel("Errore Relativo") 
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"MLP/plots/rel_error_val.png")
    plt.close()
