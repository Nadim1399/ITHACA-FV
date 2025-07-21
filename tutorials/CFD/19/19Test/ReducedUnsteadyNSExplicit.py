import numpy as np
import os
from scipy.linalg import solve
import sys
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from smithers.io.openfoam import OpenFoamHandler

class ReducedUnsteadyNSExplicit:

    def __init__(self, fluxMethod):
        self.fluxMethod = fluxMethod
        self.N_BC = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/N_BC.npy")
        self.Nphi_u = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/Nphi_u.npy")
        self.Nphi_p = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/Nphi_p.npy")
       
    def solve_online(self, vel):
        if self.fluxMethod == "inconsistent":
            self._solve_inconsistent(vel)
        elif self.fluxMethod == "consistent":
            self._solve_consistent(vel)
        else:
            print("Only the inconsistent and consistent flux methods are implemented.")
            exit(0)

    def _solve_inconsistent(self, vel):

        NUmodes = 10
        NPmodes = 10
        NSUPmodes = 0

        i = 0

        self.tstart = 0
        self.finalTime = 10

        dt = 0.005
        nu = 0.01

        C_tensor = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/C_" + str(0) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_t.npy")     

        Cf_tensor = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/Cf_" + str(0) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_t.npy")     

        RD_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/RD/RD" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        RC_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/RC/RC" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        BP_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/BP" +  "_" + str(NPmodes) + ".npy")

        P_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/P" + "_" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_" + str(NPmodes) + ".npy")

        B_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/B" + "_" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        K_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19UnsteadyNSExplicit_turb_caso7_copy/ITHACAoutput/Matrices/K" + "_" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_" + str(NPmodes) + ".npy")


        # LOAD dei file .npy
        a_o = np.load("./file_python/a_o_inc.npy")
        a_n = a_o.copy()   
        b = np.load("./file_python/b_inc.npy")
        x = np.load("./file_python/x_inc.npy")
        presidual = np.load("./file_python/presidual_inc.npy")
        RHS = np.load("./file_python/RHS_inc.npy")

        # Create and resize the solution vectors
        counter = 0
        time = self.tstart
        
        while time < self.finalTime - 0.5 * dt:
            time += dt
            counter += 1
        
        # Set the initial time
        time = self.tstart

        # Set size of online solution
        self.online_solution = [None] * (counter + 1)
        
        # Create vector to store temporal solution and save initial condition as first solution
        a_o = a_o.flatten()
        b = b.flatten()

        tmp_sol = np.zeros((int(self.Nphi_u) + int(self.Nphi_p) + 1))
        tmp_sol[0] = time
        tmp_sol[1: 1 + int(self.Nphi_u)] = a_o          
        tmp_sol[-b.shape[0]:] = b                    
        self.online_solution[0] = tmp_sol

        
        for t in range(1, len(self.online_solution)):
            time += dt
            print(f"################## time = {time} ##################")
            
            M1 = BP_matrix @ a_o * nu
            M2 = P_matrix @ a_o 

            # Pressure Poisson Equation 

            for l in range(int(self.Nphi_p)):
                cf = a_o.T @ Cf_tensor[l, :, :] @ a_o
                RHS[l] = (1 / dt) * M2[l] - cf + M1[l]

            LinSysDiv = []
            LinSysConv = []
            LinSysDiff = []

            for i in range(int(self.N_BC) + 1):
                
                filename_Div = f"./file_python/LinSysDiv_{i}.npy"
                LinSysDiv.append(np.load(filename_Div))
                
                filename_Conv = f"./file_python/LinSysConv_{i}.npy"
                LinSysConv.append(np.load(filename_Conv))
 
                filename_Diff = f"./file_python/LinSysDiff_{i}.npy"
                LinSysDiff.append(np.load(filename_Diff))

            # Boundary Term 
            RedLinSysP = LinSysDiv.copy()
            RedLinSysP[1] = RHS.copy()

            for i in range(int(self.N_BC)):
                RedLinSysP[1] += vel * ((1 / dt) * LinSysDiv[i + 1] + nu * LinSysDiff[i + 1] +
                                vel * LinSysConv[i + 1])
            
            
            presidual = RedLinSysP[0] @ x - RedLinSysP[1]
            b = np.linalg.solve(RedLinSysP[0], RedLinSysP[1])

            # Momentum Equation

            # Diffusion term 
            M5 = B_matrix @ a_o * nu
            # Pressure gradient term
            M3 = K_matrix @ b

            boundaryTerm = np.load("./file_python/boundaryTerm_inc" + ".npy")
            
            for l in range(int(self.N_BC)):
                boundaryTerm[:, l] = vel * (RD_matrix[:, l] * nu + vel * RC_matrix[:, l])

            for l in range(int(self.Nphi_u)):

                cc = a_o.T @ C_tensor[l, :, :] @ a_o
                a_n[l] = a_o[l] + (M5[l] - cc - M3[l]) * dt

                for j in range(int(self.N_BC)):
                    a_n[l] += boundaryTerm[l, j] * dt


            self.online_solution[t] = np.hstack((
                 np.array([time]),       
                 a_n.flatten(),          
                 b.flatten()             
            ))

            a_o = a_n.copy()

    def _solve_consistent(self, vel):

        NUmodes = 10
        NPmodes = 10
        NSUPmodes = 0

        i = 0

        self.tstart = 0
        self.finalTime = 10

        dt = 0.005
        nu = 0.01

        C_tensor = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/C_" + str((0)) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_t.npy")        

        Cf_tensor = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/Cf_" + str(0) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_t.npy")        

        Ci_tensor = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/Ci_" + str(0) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_t.npy")        

        RD_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/RD/RD" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        RC_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/RC/RC" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        SD_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/SD/SD" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        SC_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/SC/SC" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        W_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/W_" + str(NUmodes) + ".npy")

        BP_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/BP" +  "_" + str(NPmodes) + ".npy")

        P_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/P" + "_" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_" + str(NPmodes) + ".npy")

        B_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/B" + "_" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        K_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/K" + "_" + str(i) + "_" + str(NUmodes) + "_" + str(NSUPmodes) + "_" + str(NPmodes) + ".npy")

        I_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/I" + "_" + str(NUmodes) + ".npy")

        KF_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/KF" + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        DF_matrix = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/DF" + "_" + str(NUmodes) + "_" + str(NSUPmodes) + ".npy")

        cTotalTensor = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/ITHACAoutput/Matrices/cTotalTensor.npy")

        # Create and resize the solution vectors

        a_o = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/a_o_con" + ".npy")
        a_n = np.zeros(a_o.size)
        b = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/b_con" + ".npy")
        c_o = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/c_o_con" + ".npy")
        x = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/x_con" + ".npy")
        presidual = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/presidual_con" + ".npy")
        RHS = np.load("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/RHS_con" + ".npy")

        counter = 0
        time = self.tstart

        nut_coeffs = np.zeros(10)

        # Number of steps
        while time < self.finalTime - 0.5 * dt:
            time += dt
            counter += 1

        # Set the initial time 
        time = self.tstart

        
        # Set size of online solution 
        self.online_solution = [None] * (counter)
        
        # Create vector to store temporal solution and save initial condition as first solution
        a_o = a_o.flatten()
        b = b.flatten()
        c_o = c_o.flatten()
        
        nut_dim = len(nut_coeffs)
        tmp_sol = np.zeros((1 + int(self.Nphi_u) + int(self.Nphi_p) + int(self.Nphi_u) + nut_dim))
        tmp_sol[0] = time
        tmp_sol[1 : 1 + int(self.Nphi_u)] = a_o
        tmp_sol[1 + int(self.Nphi_u) : 1 + int(self.Nphi_u) + int(self.Nphi_p)] = b
        tmp_sol[1 + int(self.Nphi_u) + int(self.Nphi_p) : 1 + int(self.Nphi_u) + int(self.Nphi_p) + int(self.Nphi_u)] = c_o
        tmp_sol[-nut_dim:] = np.zeros(nut_dim)  

        # Modello LSTM
        # lstm_model = tf.keras.models.load_model('./Prova/trained_model.keras')
        lstm_model = tf.keras.models.load_model('./results/trained_model.keras')
        x_scaler = joblib.load("./results/x_scaler.pkl")
        y_scaler = joblib.load("./results/y_scaler.pkl")
        # x_scaler = joblib.load("./Copia/x_scaler.pkl")
        # y_scaler = joblib.load("./Copia/y_scaler.pkl")
        

        for out_iterator in range(len(self.online_solution)):
            time += dt
            print(f"################## time = {time} ##################")

            # Pressure Poisson Equation

            # Diffusion term 
            M1 = BP_matrix @ a_o * nu
            # Divergence term
            M2 = P_matrix @ a_o 

            for l in range(int(self.Nphi_p)):
                cf = c_o.T @ Cf_tensor[l, :, :] @ a_o
                RHS[l] = (1 / dt) * M2[l] - cf + M1[l]


            LinSysDiv = []
            LinSysConv = []
            LinSysDiff = []

            for i in range(int(self.N_BC) + 1):
                
                filename_Div = f"/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/LinSysDiv_{i}.npy"
                LinSysDiv.append(np.load(filename_Div))
                
                filename_Conv = f"/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/LinSysConv_{i}.npy"
                LinSysConv.append(np.load(filename_Conv))
 
                filename_Diff = f"/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Test/file_python/LinSysDiff_{i}.npy"
                LinSysDiff.append(np.load(filename_Diff))

            
            # Boundary Term 
            RedLinSysP = LinSysDiv.copy()
            RedLinSysP[1] = RHS.copy()

            for i in range(int(self.N_BC)):

                RedLinSysP[1] += vel * ((1 / dt) * LinSysDiv[i + 1] + nu * LinSysDiff[i + 1] +
                                   vel * LinSysConv[i + 1])
            

            presidual = RedLinSysP[0] @ x - RedLinSysP[1]
            b = np.linalg.solve(RedLinSysP[0], RedLinSysP[1])

            # Momentum Equation

            # Diffusion term 
            M5 = B_matrix @ a_o * nu
            # Pressure gradient term
            M3 = K_matrix @ b

            boundaryTerm = np.zeros((int(self.Nphi_u), int(self.N_BC)))

            for l in range(int(self.N_BC)):
                boundaryTerm[:, l] = vel * (RD_matrix[:, l] * nu + vel * RC_matrix[:, l])
            
            for k in range(int(self.Nphi_u)):

                cc = c_o.T @ C_tensor[k, :, :] @ a_o - nut_coeffs.T @ cTotalTensor[k, :, :] @ a_o
                a_n[k] = a_o[k] + (M5[k] - cc - M3[k]) * dt

                for l in range(int(self.N_BC)):
                    a_n[k] += boundaryTerm[k, l] * dt

            # Flux Equation

            # Mass term
            M6 = I_matrix @ a_o
            # Diffusion term
            M7 = DF_matrix @ a_o * nu
            # Pressure Gradient Term
            M8 = KF_matrix @ b[:, 0]
            # Convective term
            M9 = np.zeros((int(self.Nphi_u)))

            for k in range(int(self.Nphi_u)):
                M9 += dt * (Ci_tensor[k,:,:] @ a_o).flatten() * c_o[k]

            boundaryTermFlux = np.zeros((int(self.Nphi_u)))

            for l in range(int(self.N_BC)):
                boundaryTermFlux += vel * (SD_matrix[:, l] * nu + vel * SC_matrix[:, l])
            
            c_n = np.linalg.solve(W_matrix, M6 - M9 + dt * (-M8 + 
                  M7 + boundaryTermFlux))


            nut_dim = len(nut_coeffs)
            tmp_sol = np.zeros((1 + int(self.Nphi_u) + int(self.Nphi_p) + int(self.Nphi_u) + nut_dim))
            tmp_sol[0] = time
            tmp_sol[1 : 1 + int(self.Nphi_u)] = a_n.flatten()
            tmp_sol[1 + int(self.Nphi_u) : 1 + int(self.Nphi_u) + int(self.Nphi_p)] = b.flatten()
            tmp_sol[1 + int(self.Nphi_u) + int(self.Nphi_p) : 1 + int(self.Nphi_u) + int(self.Nphi_p) + int(self.Nphi_u)] = c_n.flatten()
            tmp_sol[-nut_dim:] = nut_coeffs.flatten()
            self.online_solution[out_iterator] = tmp_sol

            a_o = a_n.copy()
            c_o = c_n
            
            # === Calcolo nut con modello LSTM ===
            input_concat = np.concatenate((a_n.flatten(), b.flatten()), axis=0).reshape(1, -1)  
            input_scaled = x_scaler.transform(input_concat)  
            lstm_input = input_scaled[np.newaxis, :, :] 
            nut_coeffs = lstm_model.predict(lstm_input, verbose=0)[0]
            nut_coeffs = y_scaler.inverse_transform(nut_coeffs.reshape(1, -1))[0]  
            if not hasattr(self, "nut_coeffs_history"):
                 self.nut_coeffs_history = []
            self.nut_coeffs_history.append(nut_coeffs)

    def reconstruct(self, ROM_LSTM_Rec):

        os.makedirs(ROM_LSTM_Rec, exist_ok=True)

        CoeffU = []
        CoeffP = []
        CoeffNut = []
        tValues = []
        print("onl", len(self.online_solution))
        print("onl", len(self.nut_coeffs_history))

        for i in range(len(self.online_solution)):
            sol = self.online_solution[i]
            nut = self.nut_coeffs_history[i]

            if sol is None or nut is None:
                continue

            Nphi_u = self.Nphi_u[0] if isinstance(self.Nphi_u, (list, np.ndarray)) else self.Nphi_u
            Nphi_p = self.Nphi_p[0] if isinstance(self.Nphi_p, (list, np.ndarray)) else self.Nphi_p

            currentUCoeff = sol[1:1 + Nphi_u].reshape(-1, 1)
            currentPCoeff = sol[1 + Nphi_u:1 + Nphi_u + Nphi_p].reshape(-1, 1)
            currentNutCoeff = nut.reshape(-1, 1)

            CoeffU.append(currentUCoeff)
            CoeffP.append(currentPCoeff)
            CoeffNut.append(currentNutCoeff)

            time_now = sol[0] if sol[0].ndim == 0 else sol[0, 0]
            tValues.append(time_now)

        ##  LEGGERE I MODI POD  ##
        base_dir = "ITHACAoutput/POD"

        all_modes = OpenFoamHandler().read(base_dir, time_instants='all_numeric')

        u_modes = []
        p_modes = []
        nut_modes = []

        for i in range(1, 11):
            u_modes.append(all_modes[str(i)]["fields"]["U"][1])
            p_modes.append(all_modes[str(i)]["fields"]["p"][1])
            nut_modes.append(all_modes[str(i)]["fields"]["nut"][1])

        # Costruzione delle matrici finali 
        u_POD_matrix = np.stack(u_modes, axis=2)
        p_POD_matrix = np.column_stack(p_modes)
        nut_POD_matrix = np.column_stack(nut_modes)

        # Output delle dimensioni per verifica
        print("Matrice U (velocità):", u_POD_matrix.shape)
        print("Matrice P (pressione):", p_POD_matrix.shape)
        print("Matrice NUT (viscosità):", nut_POD_matrix.shape)

        CoeffU_mat = np.hstack(CoeffU)
        CoeffP_mat = np.hstack(CoeffP)
        CoeffNut_mat = np.hstack(CoeffNut)

        print("CoeffU:", CoeffU_mat.shape)
        print("Coeffp:", CoeffP_mat.shape)
        print("CoeffNut:", CoeffNut_mat.shape)

        # print("CoeffU_mat:", CoeffU_mat)
        # print("Coeffp_mat:", CoeffP_mat)
        # print("CoeffNut_mat:", CoeffNut_mat)

        np.save("Coeffs_rom/CoeffU_mat.npy", CoeffU_mat)
        np.save("Coeffs_rom/CoeffP_mat.npy", CoeffP_mat)
        np.save("Coeffs_rom/CoeffNut_mat.npy", CoeffNut_mat)

        print("u_POD_matrix", u_POD_matrix.shape)
        print("p_POD_matrix", p_POD_matrix.shape)
        print("nut_POD_matrix", nut_POD_matrix.shape)

        u_field_LSTM = u_POD_matrix @ CoeffU_mat
        p_field_LSTM = p_POD_matrix @ CoeffP_mat
        nut_field_LSTM = nut_POD_matrix @ CoeffNut_mat

        print("u_field_LSTM", u_field_LSTM.shape)
        print("p_field_LSTM", p_field_LSTM.shape)
        print("nut_field_LSTM", nut_field_LSTM.shape)

        np.save(os.path.join(ROM_LSTM_Rec, "u_field_LSTM.npy"), u_field_LSTM)
        np.save(os.path.join(ROM_LSTM_Rec, "p_field_LSTM.npy"), p_field_LSTM)
        np.save(os.path.join(ROM_LSTM_Rec, "nut_field_LSTM.npy"), nut_field_LSTM)
