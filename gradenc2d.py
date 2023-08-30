"""
Simulation of gradient-based spatial encoding.

Setup:
- Bloch simulation with gradient fields in a 2D world
- Rotating frame of ref - i.e. B0 not involved
- RF excitation ignored except for the first rep - i.e. all spins are eternally in excited state
- Relaxation effects ignored - i.e. T1=inf, T2=inf
- Phase encode direction: y-axis
"""

import numpy as np
import matplotlib.pyplot as plt


PI = np.pi
GAMMA = 42576384.74  # Gyromagnetic ratio of 1H in H20 [Hz/T]

# Object description
POS_X, POS_Y = np.meshgrid(np.linspace(-0.5, 0.5, 4), np.linspace(-0.5, 0.5, 4))  # [m]

# System config
G_PEAK = 1e-4   # [T/m]
TR = 5e-3  # [s]
T_GY_START = 1e-3   # Local time relative to the rep [s]
DURATION_GY = 1e-3  # [s]
T_GX_START = 3e-3   # Local time relative to the rep [s]
DURATION_GX = 1e-3  # [s]

# Simulation config
DT = 1e-4  # time step [s]



def init_viz(spins, gradient, full_traj, readout_traj, fig, axs):

    # Draw spin density
    spin_plots = []
    for x in range(POS_X.shape[0]):
        for y in range(POS_X.shape[1]):
            xs = [POS_X[x,y], POS_X[x,y] + 0.1*spins[x,y].real]
            ys = [POS_Y[x,y], POS_Y[x,y] + 0.1*spins[x,y].imag]
            spin_plot = axs[0].plot(xs, ys)[0]
            axs[0].scatter(POS_X[x,y], POS_Y[x,y])
            spin_plots.append(spin_plot)

    # Draw gradients
    gx, gy = gradient
    gx_viz, gy_viz = 1e2*gx, 1e2*gy
    gx_plot = axs[0].plot(
        [-0.6*np.cos(np.arctan(gx_viz)),        0.6*np.cos(np.arctan(gx_viz))], 
        [-0.6*np.sin(np.arctan(gx_viz)) - 0.75, 0.6*np.sin(np.arctan(gx_viz)) - 0.75]
        )[0]  # Gx line
    gy_plot = axs[0].plot(
        [ 0.6*np.sin(np.arctan(gy_viz)) + 0.75, -0.6*np.sin(np.arctan(gy_viz)) + 0.75], 
        [-0.6*np.cos(np.arctan(gy_viz)),         0.6*np.cos(np.arctan(gy_viz))]
        )[0]  # Gy line

    axs[0].set_xlim(-0.75, 0.9)
    axs[0].set_ylim(-0.9, 0.75)

    # Draw trajectory
    full_traj_plot = axs[1].plot([], [], ls='dashed')[0]
    readout_traj_plot = axs[1].plot([], [])[0]
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylim(-1, 1)

    return spin_plots, gx_plot, gy_plot, full_traj_plot, readout_traj_plot


def update_viz(
    spins, gradient, full_traj, readout_traj, 
    fig, axs, spin_plots, gx_plot, gy_plot, full_traj_plot, readout_traj_plot
    ):

    # Update spins
    for x in range(POS_X.shape[0]):
        for y in range(POS_X.shape[1]):
            xs = [POS_X[x,y], POS_X[x,y] + 0.1*spins[x,y].real]
            ys = [POS_Y[x,y], POS_Y[x,y] + 0.1*spins[x,y].imag]
            spin_plots[x*POS_X.shape[0] + y].set_xdata(xs)
            spin_plots[x*POS_X.shape[0] + y].set_ydata(ys)
    
    # Update gradients
    gx, gy = gradient
    gx_viz, gy_viz = 2e3*gx, 2e3*gy
    gx_plot.set_xdata([-0.6*np.cos(np.arctan(gx_viz)) ,        0.6*np.cos(np.arctan(gx_viz))])
    gx_plot.set_ydata([-0.6*np.sin(np.arctan(gx_viz)) - 0.75,  0.6*np.sin(np.arctan(gx_viz)) - 0.75])
    gy_plot.set_xdata([ 0.6*np.sin(np.arctan(gy_viz)) + 0.75, -0.6*np.sin(np.arctan(gy_viz)) + 0.75])
    gy_plot.set_ydata([-0.6*np.cos(np.arctan(gy_viz)),         0.6*np.cos(np.arctan(gy_viz))])
    
    # Update trajectory
    if len(full_traj) > 0:
        full_traj = np.array(full_traj)
        full_traj_plot.set_xdata(full_traj[:,0])
        full_traj_plot.set_ydata(full_traj[:,1])
    if len(readout_traj) > 0:
        readout_traj = np.array(readout_traj)
        readout_traj_plot.set_xdata(readout_traj[:,0])
        readout_traj_plot.set_ydata(readout_traj[:,1])

    fig.canvas.draw()
    fig.canvas.flush_events() 


def update_spins(spins, gradients):
    gx, gy = gradients
    for x in range(POS_X.shape[0]):
        for y in range(POS_X.shape[1]):
            spins[x, y] *= np.exp(1j * GAMMA * (gx*POS_X[x, y] + gy*POS_Y[x, y]) * DT)
    return spins


def update_trajectory(traj, gradient_history):
    gradient_history = np.array(gradient_history)
    gx_history = gradient_history[:,0]
    gy_history = gradient_history[:,1]
    kx = (GAMMA / (2*PI)) * np.sum(np.array(gx_history) * DT)
    ky = (GAMMA / (2*PI)) * np.sum(np.array(gy_history) * DT)
    traj.append((kx, ky))
    return traj


def main():
    
    spins = np.full((4,4), 1+0j, dtype=np.complex64)

    num_freq_encodes = spins.shape[0]
    num_phase_encodes = spins.shape[1]
    full_traj = []
    readout_traj = []

    # Init viz
    fig, axs = plt.subplots(1,2,figsize=(12, 5))
    spin_plots, gx_plot, gy_plot, full_traj_plot, readout_traj_plot = init_viz(spins, [0,0], full_traj, readout_traj, fig, axs)
    plt.ion()
    plt.show()
    
    # Run gradenc
    gradient_history = []
    for rep in range(num_phase_encodes + 1):

        # Local time inside the rep
        t_vec = np.arange(0, TR, DT)

        for t in t_vec:                    

            # PE gradient
            if t > T_GY_START and t < (T_GY_START + DURATION_GY):
                # If first rep, move kspace position from origin to bottom-left corner
                if rep == 0: gy = -0.5 * G_PEAK
                else:        gy = (1/POS_X.shape[1]) * G_PEAK
                full_traj = update_trajectory(full_traj, gradient_history)

            # FE gradient and readout
            elif t > T_GX_START and t < (T_GX_START + DURATION_GX):
                # If first rep, move kspace position from origin to bottom-left corner
                if rep == 0: gx = -0.5 * G_PEAK
                else:        gx = G_PEAK
                full_traj = update_trajectory(full_traj, gradient_history)
                readout_traj = update_trajectory(readout_traj, gradient_history)                
            
            # All other time periods in the rep
            else:
                gx, gy = 0, 0

            gradient_history.append([gx, gy])
            spins = update_spins(spins, [gx, gy])            
            update_viz(
                spins, [gx, gy], full_traj, readout_traj, 
                fig, axs, spin_plots, gx_plot, gy_plot, full_traj_plot, readout_traj_plot
                )


if __name__ == '__main__':
    main()