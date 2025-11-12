import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern",
})

# ERRORI LSTM #
# error_U = np.load('error_U.npy')
# error_p = np.load('error_p.npy')
# error_nut = np.load('error_nut.npy')

# ERRORI MLP #
error_U = np.load('./MLP/error_U.npy')
error_p = np.load('./MLP/error_p.npy')
error_nut = np.load('./MLP/error_nut.npy')

# ERRORI TANSFORMER #
# error_U = np.load('./Tr/error_U.npy')
# error_p = np.load('./Tr/error_p.npy')
# error_nut = np.load('./Tr/error_nut.npy')

x = np.linspace(1, len(error_U), len(error_U)) 

def plot_error(x, error, label, filename, title):
    plt.figure() 

    plt.plot(x, error, marker='o', markevery=200, label=label, linewidth=2)

    plt.xlabel("Time Step", fontsize=25)
    plt.ylabel("Relative L2 Error", fontsize=25)
    plt.title(title, fontsize=25)
    plt.tick_params(axis='both', labelsize=24)

    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', color='gray', linewidth=0.5)
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))

    plt.legend(fontsize=20)
    plt.tight_layout() 
    plt.savefig(filename)
    plt.close() 
    print(f"Saved: {filename}")

###### LSTM ########
# plot_error(x, error_U, "Error_U", "error_U_plot.png", "LSTM Relative L2 Error - U")
# plot_error(x, error_p, "Error_p", "error_p_plot.png", "LSTM Relative L2 Error - p")
# plot_error(x, error_nut, "Error_nut", "error_nut_plot.png", "LSTM Relative L2 Error - nut")

###### MLP ########
# plot_error(x, error_U, "Error_MLP_U", "error_MLP_U_plot.png", "MLP Relative L2 Error - U")
# plot_error(x, error_p, "Error_MLP_p", "error_MLP_p_plot.png", "MLP Relative L2 Error - p")
# plot_error(x, error_nut, "Error_MLP_nut", "error_MLP_nut_plot.png", "MLP Relative L2 Error - nut")

###### TRANSFORMER ########
plot_error(x, error_U, "Error_Tr_U", "error_Tr_U_plot.png", "Transformer Relative L2 Error - U")
plot_error(x, error_p, "Error_Tr_p", "error_Tr_p_plot.png", "Transformer Relative L2 Error - p")
plot_error(x, error_nut, "Error_Tr_nut", "error_Tr_nut_plot.png", "Transformer Relative L2 Error - nut")



###### LSTM ########
# plot_error(x, error_U, "Error_U", "error_U_plot.pdf", "LSTM  Relative L2 Error - U")
# plot_error(x, error_p, "Error_p", "error_p_plot.pdf", "LSTM  Relative L2 Error - p")
# plot_error(x, error_nut, "Error_nut", "error_nut_plot.pdf", "LSTM  Relative L2 Error - nut")

###### MLP ########
# plot_error(x, error_U, "Error_MLP_U", "error_MLP_U_plot.pdf", "MLP Relative L2 Error - U")
# plot_error(x, error_p, "Error_MLP_p", "error_MLP_p_plot.pdf", "MLP Relative L2 Error - p")
# plot_error(x, error_nut, "Error_MLP_nut", "error_MLP_nut_plot.pdf", "MLP Relative L2 Error - nut")

###### TRANSFORMER ########
plot_error(x, error_U, "Error_Tr_U", "error_Tr_U_plot.pdf", "Transformer Relative L2 Error - U")
plot_error(x, error_p, "Error_Tr_p", "error_Tr_p_plot.pdf", "Transformer Relative L2 Error - p")
plot_error(x, error_nut, "Error_Tr_nut", "error_Tr_nut_plot.pdf", "Transformer Relative L2 Error - nut")
