"""
Codice per vedere le differenze di
visualizzazione tra diverse colormap
"""
import numpy as np
import matplotlib.pyplot as plt

# ============= Data =============
x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)
X, Y = np.meshgrid(x, y)

Z1 = np.exp(-(X**2 + Y**2)/2) * (np.sin(3 * X) + np.cos(3 * Y) + 2)
Z2 = X**2 - Y**2
Z3 = X + Y

# ============= Plot =============

def plot_comparison(Z, cmap_wrong, cmap_correct):
    '''
    Funzione per plottare due grafici a confronto

    Parameters
    ----------
    Z : np.array
        Matrice di dati.
    cmap_wrong : str
        Nome della colormap sbagliata.
    cmap_correct : str
        Nome della colormap corretta.        
    '''
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    p1 = ax1.pcolormesh(X, Y, Z, cmap=cmap_wrong)
    ax1.set_title(cmap_wrong)
    fig.colorbar(p1, ax=ax1)

    ax2 = axes[1]
    p2 = ax2.pcolormesh(X, Y, Z, cmap=cmap_correct)
    ax2.set_title(cmap_correct)
    fig.colorbar(p2, ax=ax2)

    plt.show()

plot_comparison(Z1, 'coolwarm', 'viridis')


plot_comparison(Z2, 'inferno',  'RdBu'   )
plot_comparison(Z3, 'jet',      'plasma' )

"""fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z1)
plt.show()"""