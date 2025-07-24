import numpy as np
import matplotlib.pyplot as plt

# ERRORI LSTM #
# error_U = np.load('error_U.npy')
# error_p = np.load('error_p.npy')
# error_nut = np.load('error_nut.npy')

# ERRORI MLP #
# error_U = np.load('./MLP/error_U.npy')
# error_p = np.load('./MLP/error_p.npy')
# error_nut = np.load('./MLP/error_nut.npy')

# ERRORI TANSFORMER #
error_U = np.load('./Tr/error_U.npy')
error_p = np.load('./Tr/error_p.npy')
error_nut = np.load('./Tr/error_nut.npy')

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

###### LSTM ########
# plot_error(x, error_U, "Error_U", "error_U_plot.png", "Relative L2 Error - U")
# plot_error(x, error_p, "Error_p", "error_p_plot.png", "Relative L2 Error - p")
# plot_error(x, error_nut, "Error_nut", "error_nut_plot.png", "Relative L2 Error - nut")

###### MLP ########
# plot_error(x, error_U, "Error_MLP_U", "error_MLP_U_plot.png", "Relative L2 Error - U")
# plot_error(x, error_p, "Error_MLP_p", "error_MLP_p_plot.png", "Relative L2 Error - p")
# plot_error(x, error_nut, "Error_MLP_nut", "error_MLP_nut_plot.png", "Relative L2 Error - nut")

###### TRANSFORMER ########
plot_error(x, error_U, "Error_Tr_U", "error_Tr_U_plot.png", "Relative L2 Error - U")
plot_error(x, error_p, "Error_Tr_p", "error_Tr_p_plot.png", "Relative L2 Error - p")
plot_error(x, error_nut, "Error_Tr_nut", "error_Tr_nut_plot.png", "Relative L2 Error - nut")
