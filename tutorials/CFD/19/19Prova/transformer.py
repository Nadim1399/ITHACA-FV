import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
import joblib

# === CREA CARTELLE RISULTATI ===
os.makedirs("Transformer/plots", exist_ok=True)

# ==== PARAMETRI ====
lookback = 15
epochs = 1000
batch_size = 64

# ==== CARICAMENTO DATI POD ====
coeffs_u = np.load("./Coeffs/u_coeffs.npy")
coeffs_p = np.load("./Coeffs/p_coeffs.npy")
coeffs_nut = np.load("./Coeffs/nut_coeffs.npy")

# ==== DIVISIONE TRAIN/VALID ====
u_coeffs_1800 = coeffs_u[:, :1800]
p_coeffs_1800 = coeffs_p[:, :1800]
nut_coeffs_1800 = coeffs_nut[:, :1800]

u_coeffs_last201 = coeffs_u[:, -201:]
p_coeffs_last201 = coeffs_p[:, -201:]
nut_coeffs_last201 = coeffs_nut[:, -201:]

X_all = np.hstack([u_coeffs_1800.T, p_coeffs_1800.T])
Y_all = nut_coeffs_1800.T

X_val_raw = np.hstack([u_coeffs_last201.T, p_coeffs_last201.T])
Y_val_raw = nut_coeffs_last201.T

# ==== NORMALIZZAZIONE ====
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_all = x_scaler.fit_transform(X_all)
Y_all = y_scaler.fit_transform(Y_all)

X_val = x_scaler.transform(X_val_raw)
Y_val = y_scaler.transform(Y_val_raw)

joblib.dump(x_scaler, "Transformer/x_scaler.pkl")
joblib.dump(y_scaler, "Transformer/y_scaler.pkl")

# ==== CREAZIONE SEQUENZE ====
def create_sequences(X, Y, lookback, step=1):
    X_seq, Y_seq = [], []
    for i in range(0, len(X) - lookback, step):
        X_seq.append(X[i:i+lookback])
        Y_seq.append(Y[i+lookback])
    return np.array(X_seq), np.array(Y_seq)

X_train_seq, Y_train_seq = create_sequences(X_all, Y_all, lookback)
X_val_seq, Y_val_seq = create_sequences(X_val, Y_val, lookback)

# ==== DEFINIZIONE BLOCCO TRANSFORMER ====
def transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn = Dropout(dropout)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn)
    
    ff = Dense(ff_dim, activation='relu')(out1)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    return LayerNormalization(epsilon=1e-6)(out1 + ff)

# ==== COSTRUZIONE MODELLO ====
input_shape = (lookback, X_train_seq.shape[2])
inputs = Input(shape=input_shape)

x = transformer_encoder(inputs)
x = transformer_encoder(x)

x = Dense(64, activation='relu')(x)
x = Dense(Y_train_seq.shape[1])(x[:, -1, :])  

model = Model(inputs, x)
model.compile(optimizer=Adam(9e-5), loss='mse')
model.summary()

# ==== TRAINING ====
history = model.fit(X_train_seq, Y_train_seq, validation_data=(X_val_seq, Y_val_seq), epochs=epochs,
                    batch_size=batch_size, verbose=1)

# ==== PREDIZIONE ====
Y_train_pred = model.predict(X_train_seq)
Y_val_pred = model.predict(X_val_seq)

Y_train_pred_orig = y_scaler.inverse_transform(Y_train_pred)
Y_train_true_orig = y_scaler.inverse_transform(Y_train_seq)

Y_val_pred_orig = y_scaler.inverse_transform(Y_val_pred)
Y_val_true_orig = y_scaler.inverse_transform(Y_val_seq)

# ==== SALVATAGGI ====
model.save("Transformer/transformer_model.keras")
np.save("Transformer/Y_val_pred.npy", Y_val_pred_orig)
np.save("Transformer/Y_val_true.npy", Y_val_true_orig)
np.save("Transformer/training_history.npy", history.history)

# ==== PLOT LOSS ====
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training History - Transformer")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("Transformer/plots/training_history_transformer.png")
plt.close()

# ==== METRICHE PER MODO ====
errors = {'mse': [], 'mae': []}
for i in range(Y_val_true_orig.shape[1]):
    true_vals = Y_val_true_orig[:, i]
    pred_vals = Y_val_pred_orig[:, i]
    errors['mse'].append(mean_squared_error(true_vals, pred_vals))
    errors['mae'].append(mean_absolute_error(true_vals, pred_vals))

errors_array = np.column_stack((errors['mse'], errors['mae']))
np.savetxt("Transformer/validation_errors.csv", errors_array, delimiter=",", header="mse,mae", comments='')

# ==== ERRORE PER SNAPSHOT ====
snapshot_mse = np.mean((Y_val_true_orig - Y_val_pred_orig)**2, axis=1)
np.save("Transformer/snapshot_mse.npy", snapshot_mse)

plt.figure(figsize=(10, 4))
plt.plot(snapshot_mse)
plt.title("Snapshot-wise MSE (Validation Set)")
plt.xlabel("Time step")
plt.ylabel("MSE")
plt.grid()
plt.tight_layout()
plt.savefig("Transformer/plots/snapshot_mse.png")
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
    plt.savefig(f"Transformer/plots/abs_error_train.png")
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
    plt.savefig(f"Transformer/plots/rel_error_val.png")
    plt.close()