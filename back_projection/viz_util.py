from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

test_frustum_bounding = np.array([
    # x, y, z, 1
    [[4.1416], [1.3231], [1.2873], [1.0000]],
    [[3.7728], [1.5169], [1.3048], [1.0000]],
    [[3.8193], [1.6308], [1.0239], [1.0000]],
    [[4.1880], [1.4370], [1.0063], [1.0000]],
    [[4.0026], [-2.9551], [1.0370], [1.0000]],
    [[0.3151], [-1.0170], [1.2127], [1.0000]],
    [[0.7797], [0.1216], [-1.5969], [1.0000]],
    [[4.4672], [-1.8165], [-1.7726], [1.0000]]])

def viz_frustum_bounds(frustum_bounds):
    """ Visualize the frustum bounds computed by projection::compute_frustum_bounds
    :param frustum_bounds: a numpy array of shape (8, 4, 1)
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata = np.array([])
    ydata = np.array([])
    zdata = np.array([])
    for pos in frustum_bounds:
        xdata = np.append(xdata, pos[0])
        ydata = np.append(ydata, pos[1])
        zdata = np.append(zdata, pos[2])

    # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.scatter3D(xdata, ydata, zdata)
    plt.show()

if __name__ == "__main__":
    # demo
    viz_frustum_bounds(test_frustum_bounding)