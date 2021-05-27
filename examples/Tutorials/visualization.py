import matplotlib.pyplot as plt
import numpy as np

def plot_batch_stack(Recomendations, wavelength, TARGET, batch_size, epochs, env):
    
    nrows = epochs//2
    fig, axes = plt.subplots(figsize=(10,15), nrows=nrows, ncols=2)
    fontsize = 16

    total_steps = Recomendations.shape[0]
    
    # formatting loop
    for i, ax in enumerate(axes.flatten()):
        # ticks
        ax.tick_params(direction='out', width=2, length=8)
        ax.tick_params(direction='out', which='minor', width=1, length=5)
        ax.set_yticks([])
        
        if i in [epochs-2, epochs-1]:
            ax.set_xticks(np.arange(300, 701, 100))
            ax.tick_params(axis='x', labelsize=fontsize+2)
            ax.set_xlabel("Wavelength", fontsize=fontsize+6)
        else:
            ax.set_xticks([])
     
    # actually plotting loop
    for i, ax in enumerate(axes.flatten()):
        # target
        ax.plot(wavelength, TARGET, 'k-', linewidth=3, label="Target")
        # get batch
        for action in Recomendations[i]:
            ax.plot(wavelength, env.spectra_from_conc(action), '-',
                     c=action, alpha=0.4)
        # setting legend as batch label
        legend = ax.legend([f'Batch {i+1}'], fontsize=fontsize+4, handlelength=0,
                           handletextpad=0, fancybox=True,)
        for item in legend.legendHandles:
            item.set_visible(False)

    plt.subplots_adjust(hspace=0.0, wspace=0.05)
    plt.show()

def plot_batch(Results, actions, wavelength, target, batch_num):
    
    fig, ax = plt.subplots(figsize=(8,6), nrows=1, ncols=1)
    fontsize = 16
    
    # formatting loop
    ax.tick_params(direction='out', width=2, length=8)
    ax.tick_params(direction='out', which='minor', width=1, length=5)
    ax.set_yticks([])
        
    ax.set_xticks(np.arange(300, 701, 100))
    ax.tick_params(axis='x', labelsize=fontsize+2)
    ax.set_xlabel("Wavelength", fontsize=fontsize+6)

    # target
    ax.plot(wavelength, target, 'k-', linewidth=3, label="Target")
    for i, spectrum in enumerate(Results):
        ax.plot(wavelength, spectrum, '-', c=actions[i], alpha=0.4)
    # setting legend as batch label
    legend = ax.legend([f'Batch {batch_num}'], fontsize=fontsize+4, handlelength=0,
                        handletextpad=0, fancybox=True,)
    for item in legend.legendHandles:
        item.set_visible(False)

    plt.subplots_adjust(hspace=0.0, wspace=0.05)
    plt.show()

def plot_best_spectrum(X, Y, batch_size, wavelength, TARGET, env):
    best_idx = np.argmax(Y)
    batch_num = best_idx // batch_size
    action = X[batch_num, best_idx%batch_size, :]
    best_spectrum = env.spectra_from_conc(action)

    best_spectrum = best_spectrum - np.min(best_spectrum)
    best_spectrum = best_spectrum / np.max(best_spectrum)

    # plot target spectra
    fig, ax = plt.subplots(figsize=(6,4))
    plt.plot(wavelength, TARGET, 'k-', linewidth=1)
    plt.plot(wavelength, best_spectrum, linewidth=3)
    # formatting
    ax.tick_params(direction='out', width=2, length=8)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Best spectrum: {best_idx}th try")
    plt.show()