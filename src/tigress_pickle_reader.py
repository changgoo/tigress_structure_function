import pandas as pd
import numpy as np
import xarray as xr

import astropy.constants as ac
import astropy.units as au

# define the conversion factor from pc/(km/s) to Myr
tMyr = (ac.pc / (au.km / au.s)).to("Myr").value


class TigressPickle(object):
    def __init__(self, fname, axis="z", verbose=False):
        self.read_tigress_pickle(fname, axis=axis)
        self.verbose = verbose

    def __repr__(self):
        out = "TIGRESS pickle output:"
        out += f" ({self.domain["Nx"]}x{self.domain["Ny"]}) data"
        out += f" at time={self.time}"
        if self.verbose:
            out += "\n domain info (length in pc): \n"
            for i, k in enumerate(self.domain):
                out += f"  {k} = {self.domain[k]}"
                if (i + 1) % 4 == 0:
                    out += "\n"
            out += "\n surface density (in Msun/pc^2):"
            out += f"{self.surf.flat[0]}...{self.surf.flat[-1]}"
        return out

    def read_tigress_pickle(self, fname, axis="z"):
        """Read TIGRESS pickle file

        Parameters
        ----------
        fname : str
            pickle file name
        axis : str
            axis along which the data is projected (default: "z")

        Members
        -------
        surf : 2D array
            projected data
        domain : dict
            domain information
        coords : dict
            cell-centered coordinates
        coords_f : dict
            face-centered coordinates
        time : float
            simulation time in the code unit
        tMyr : float
            simulation time in Myr

        Returns
        -------
        None

        Examples
        --------
        >>> fname = "R8_8pc_NCR.full.z.p"
        >>> tigress = TigressPickle(fname)
        >>> print(tigress)
        TIGRESS pickle output: (2048x2048) data at time=290.0

        """
        self.fname = fname
        self.axis = axis

        # read the original pickle file using pandas
        data = pd.read_pickle(fname)

        # extract time information
        self.tMyr = data["time"]  # in Myr
        self.time = self.tMyr / tMyr  # in the code unit [pc/(km/s)]

        # extrad surface density
        self.surf = data[axis]["data"]  # in Msun/pc^2
        # convert surface density to column density (cm^-2)
        self.NH = (self.surf * ac.M_sun / ac.pc**2 / (1.4271 * ac.m_p)).to("cm-2").value

        # extract domain information
        xmin, xmax, ymin, ymax = data[axis]["bounds"]
        self.domain = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        Lx = xmax - xmin
        Ly = ymax - ymin
        Ny, Nx = self.surf.shape
        dx = Lx / Nx
        dy = Ly / Nx
        self.domain.update(dict(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, dx=dx, dy=dy))
        xfc = np.linspace(xmin, xmax, Nx + 1)
        yfc = np.linspace(ymin, ymax, Ny + 1)
        xcc = 0.5 * (xfc[1:] + xfc[:-1])
        ycc = 0.5 * (yfc[1:] + yfc[:-1])

        # set coordinates
        self.coords = dict(y=ycc, x=xcc)
        self.coords_f = dict(yfc=yfc, xfc=xfc)

    def to_xarray(self):
        return xr.DataArray(self.surf, coords=self.coords)

    def get_yshear(self, qshear=1, Omega=0.028):
        Lx = self.domain["Lx"]
        time = self.time
        return qshear * Omega * Lx * time
