import xarray as xr
import numpy as np
from scipy.ndimage import shift


def expand_x(data, domain, ysh=0):
    """function to expand original data from -Lx/2 < x < Lx/2 to -3*Lx/2 < x < 3*Lx/2

    Parameters
    ----------
    data : xarray.Dataset in the original domain

    domain : dict containing Lx, dy

    Returns
    -------
    xarray.Dataset with extended x-domain (tripled)
    """
    # get domain information
    Lx = domain["Lx"]
    dy = domain["dy"]

    # exapnd in x assuming periodic BC
    data_left = data.copy(deep=True).assign_coords(x=data.coords["x"] - Lx)
    data_right = data.copy(deep=True).assign_coords(x=data.coords["x"] + Lx)

    # exapnd in x assuming shear-periodic BC
    if ysh != 0:
        # get shear related parameters
        dims = data.dims
        ndims = 2
        # apply y-shift due to shear for each variable

        # shift for the left
        shifts = np.zeros(ndims)
        shifts[0] = ysh / dy
        shifted_L = shift(data_left.data, shifts, mode="grid-wrap", order=1)
        # shift for the right
        shifts = np.zeros(ndims)
        shifts[0] = -ysh / dy
        shifted_R = shift(data_right.data, shifts, mode="grid-wrap", order=1)
        # add shear velocity
        # if var == "vy":
        #     shifted_L += vsh
        #     shifted_R -= vsh
        # update L/R data
        data_left.data = shifted_L
        data_right.data = shifted_R
    return xr.concat([data_left, data, data_right], dim="x")


def expand_y(data, domain):
    """function to expand original data from -Ly/2 < y < Ly/2 to -3*Ly/2 < y < 3*Ly/2

    Parameters
    ----------
    data : xarray.Dataset in the original domain

    domain : dict containing Ly

    Returns
    -------
    xarray.Dataset with extended y-domain (tripled)
    """
    # get domain information
    Ly = domain["Ly"]

    # exapnd in x assuming periodic BC
    data_bot = data.copy(deep=True).assign_coords(y=data.coords["y"] - Ly)
    data_top = data.copy(deep=True).assign_coords(y=data.coords["y"] + Ly)

    return xr.concat([data_bot, data, data_top], dim="y")


def expand_xy(data, domain, ysh=0):
    """Triple XY domain

    Parameters
    ----------
    sim : pyathena.LoadSim class (must called load_vtk to set `ds`)

    data : xarray.Dataset in the original domain returned by ds.get_field()

    Returns
    -------
    xarray.Dataset with extended xy-domain (tripled)

    Example
    -------
    >>> from pyathena.tigress_ncr.load_sim_tigress_ncr import LoadSimTIGRESSNCR
    >>> s = LoadSimTIGRESSNCR("/tigress/changgoo/TIGRESS-NCR/R8_8pc_NCR.full.xy2048.eps0.0",verbose=True)
    >>> ds = s.load_vtk(290)
    >>> data = ds.get_field(["nH","T","vy"])
    >>> data_exp = expand_xy(s,data.sel(z=0,method="nearest")) # expanded 2D slices
    >>> data_exp["vy"].plot(**s.dfi["vy"]["imshow_args"])
    """
    return expand_y(expand_x(data, domain, ysh=ysh), domain)
