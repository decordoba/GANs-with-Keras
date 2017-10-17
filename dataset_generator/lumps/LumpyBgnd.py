import numpy as np
import random


def lumpy_backround(dim=(64, 64), nbar=200, dc=10, lump_function="GaussLmp", pars=(1, 10), discretize_lumps_positions=False, rng=None):
    """
    : param dim: Output image dimensions. Can be 2D tuple or int (will convert it to a square image)
    : param nbar: Mean number of lumps in the image
    : param dc: DC offset of output image
    : param lump_function: Either 'GaussLmp' or 'CircLmp', for Gaussian or Circular lumps
    : param pars: (magnitude, stddev) for 'GaussLmp'
                  (magnitude, radius) for 'CircLmp'
    : param discretize_lumps_positions: If True, all positions are ints, else, they can be floats
    : return: (image, n, lumps_pos)
               image: Generated image with lumps
               N: Number of lumps
               lumps_pos: Position of every lump in image
    """

    # Assume square image if dim is an integer
    if isinstance(dim, int):
      dim = (dim, dim)

    # Initialize values that will be returned
    image = dc * np.ones(dim)
    n = np.random.poisson(nbar)
    lumps_pos = []

    for i in range(n):
        # Random position of lump, uniform throughout image
        if discretize_lumps_positions:
            pos = [int(np.random.rand() * d) for d in dim]
        else:
            pos = [np.random.rand() * d for d in dim]
        pos = tuple(pos)
        lumps_pos.append(pos)
        
        # Set up a grid of points
        x, y = np.meshgrid(np.array(range(dim[0])) - pos[0],
                           np.array(range(dim[1])) - pos[1])

        # Generate a lump centered at pos
        if lump_function == "GaussLmp":
            lump = pars[0] * np.exp(-0.5 * (x ** 2 + y ** 2) / (pars[1] ** 2))
        elif lump_function == "CircLmp":
            lump = pars[0] * ((x ** 2 + y ** 2) <= (pars[1] ** 2))
        else:
            raise Exception("Unknown lump function '{}'".format(lump_function))
        
        # Add lump to the image
        image = image + lump

    # Rescale image to range rng
    if rng is not None:
        # If range is int, assume rng gomes from 0
        if isinstance(rng, int):
            rng = (0, rng)
        min_v = image.min()
        max_v = image.max()
        if min_v == max_v:  # Avoid dividing by zero
            image = rng[0] * np.ones(dim)
        else:
            image = (image - min_v) / (max_v - min_v) * (rng[1] - rng[0]) + rng[0]

    return image, n, lumps_pos

def create_lumps_pos_matrix(lumps_pos, dim=(64, 64), discrete_lumps_positions=False):
    """
    :param dim: Output image dimensions. Can be 2D tuple or int (will convert it to a square image)
    :param lumps_pos: Position of every lump in image
    :param discrete_lumps_positions: # If True, all positions will be discretized (floored), else, they can be floats
    :return: 
    """
    # Assume square image if dim is an integer
    if isinstance(dim, int):
        dim = (dim, dim)
    
    # Put a 1 in the matrix in all the lump positions.
    # If the position is not discrete, split this 1 among the discrete positions in image
    image = np.zeros(dim)
    for pos in lumps_pos:
        if discrete_lumps_positions:
            image[int(pos[1]), int(pos[0])] += 1
        else:
            x = pos[0]
            xl_pos = int(x)
            xh_pos = int(x) + 1
            xl = x - xl_pos
            xh = xh_pos - x
            y = pos[1]
            yl_pos = int(y)
            yh_pos = int(y) + 1
            yl = y - yl_pos
            yh = yh_pos - y
            image[yl_pos, xl_pos] += yh * xh
            if xh_pos < dim[0]:
                image[yl_pos, xh_pos] += yh * xl
            if yh_pos < dim[1]:
                image[yh_pos, xl_pos] += yl * xh
            if xh_pos < dim[0] and yh_pos < dim[1]:
                image[yh_pos, xh_pos] += yl * xl
    return image
    

if __name__ == "__main__":
    dim = 5
    nbar = 2
    dc = 0
    lump_function = "GaussLmp"
    pars = (1, 1)
    discrete_lumps = True
    rng = (0, 255)
    image, n, lumps_pos = lumpy_backround(dim=dim, nbar=nbar, dc=dc, lump_function=lump_function, pars=pars,
                                          discretize_lumps_positions=discrete_lumps, rng=rng)
    
    
    image_pos = create_lumps_pos_matrix(dim=dim, lumps_pos=lumps_pos)
    
    print("N:", n)
    print("Lumps position:", lumps_pos)
    print("Image:\n", image)
    print("Position matrix:\n", image_pos)
    