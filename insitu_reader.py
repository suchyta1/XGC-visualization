"""Module of the XGC1 loader for regerating general plots using ADIOS2
Some parts are taken from Michael's xgc.py which is taken from Loic's load_XGC_local for BES.
It reads the data from the simulation especially 1D results and other small data output.

TODO
3D data are loaded only when it is specified.
"""

import numpy as np
import os
from matplotlib.tri import Triangulation
import adios2
import matplotlib.pyplot as plt
from scipy.io import matlab
from scipy.optimize import curve_fit
from scipy.special import erfc
import scipy.sparse as sp

import re
import kittie_common


class xgc1(object):
    
    def __init__(self, datadir, options):
        self.datadir = datadir
        self.options = options
        self.outdir = "plots"

        self.dataunits = self.BaseData(os.path.join(self.datadir, "xgc.units.bp"), "output.units")
        self.f0mesh = self.BaseDate(os.path.join(self.datadir, "xgc.f0.mesh.bp"), "diagnosis.f0.mesh")
        self.mesh = self.BaseDate(os.path.join(self.datadir, "xgc.mesh.bp"), "diagnosis.mesh")
        self.data3D = self.BaseData(os.path.join(self.datadir, "xgc.3d.bp"), "field3D")

        if options['turbulence intensity']:
            self.data3D.on = True
            self.data3D.variables += ["dpot"]
            self.DefaultOption('turbulence intensity', 'outdir', 'TurbulenceIntensity', self.data3D)
            self.DefaultOption('turbulence intensity', 'ext', 'svg', self.data3D)
            self.DefaultOption('turbulence intensity', 'psirange', [0.17, 0.4], self.data3D)
            self.DefaultOption('turbulence intensity', 'file-per-step', False, self.data3D)

            self.dataunits.on = True
            self.dataunits.variable += ["sml_dt", "diag_1d_period"]
            self.f0mesh.on = True
            self.f0mesh.variable += ["f0_T_ev"]
            self.mesh.on = True
            self.mesh.variable += ["node_vol", "psi"]
            self.diag_3d_period = self.GetText("input", "^\s*diag_3d_period\s*=(\d*).*$")

        
        if self.dataunit.on:
            self.SingleRead(self.dataunits)
            self.dataunits.psi_x = GetText("units.m", "^\s*psi_x\s*=(.*)\s*;\s*$")

        if self.f0mesh.on:
            self.SingleRead(self.f0mesh)

        if self.mesh.on:
            self.SingleRead(self.mesh)

        if self.data3D.on:
            self.data3D.init()


    def OptionDefault(self, name, key, default, data):
        if key in self.options[name]:
            setattr(self, key, self.options[name][key])
        else:
            setattr(self, key, default)


    def GetText(self, filename, searchpattern):
        infile = os.path.join(self.datadir, filename)
        with open(infile) as f:
            intxt = f.read()
        pattern = re.compile(searchpattern, re.MULTILINE)
        matches = pattern.findall(intxt)
        if len(matches) > 0:
            return int(matches[-1])
        else:
            return None


    def SingleRead(self, data):
        self.data.init()
        while not self.data.NewDataCheck():
            pass
        self.data.GetData()
        self.data.Stop()


    def NotDone(self):
        return self.data3D.on


    def MakePlots(self):
        new3D = False

        if self.data3D.on and self.data3D.NewDataCheck():
            self.data3D.GetData()
            new3D = True

        if options['turbulence intensity'] and new3D:
            self.TurbulenceIntensity()


    def TurbulenceIntensity(self):

        if self.data3D.StepNum == 0:

            psimesh = self.mesh.psi / self.dataunits.psi_x
            mask = np.nonzero( (psimesh > self.data3D.psirange[0]) & (psimesh < self.data3D.psirange[1]) )

            #self.en = np.empty( shape=(0, 0) )
            self.enp = np.empty( shape=(0, 0) )
            self.enn = np.empty( shape=(np.sum(mask), 0))
            self.TurbTime = []

            gs = gridspec.GridSpec(1, 1)
            fig = plt.figure()
            ax = fig.add_subplot(gs[0, 0])


	var1 = self.data3D.dpot - np.mean(self.data3D.dpot, axis=0)
	var1 = var1 / self.f0mesh.T0
	varsqr = var1 * var1

        # Not using yet
	#s = np.mean(varsqr * self.mesh.node_vol) / np.mean(self.mesh.node_vol)
	#en = np.append(en, s)

	# partial sum
	sp = np.mean(varsqr[:, mask] * self.mesh.node_vol[mask]) / np.mean(self.mesh.node_vol[mask])
	self.enp = np.append(self.enp, sp)

        # spectral
        vft = np.abs(np.fft.fft(var1, axis=0))**2
        sn = np.mean(vft[:, mask] * self.mesh.node_vol[mask], axis=1) / np.mean(self.mesh.node_vol[mask])
        sn = sn[:, np.newaxis]
        enn = np.append(enn, sn, axis=1)

        if self.diag_3d_period == None:
            self.diag_3d_period = self.dataunits.diag_1d_period
        self.TurbTime += [(sef.diag_3d_period * self.dataunits.sml_dt) * (self.data3D.StepNumber + 1) * 1E3]  # ms
      
        ax.semilogy(self.TurbTime, np.sqrt(enp))
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('$\sqrt{<(\phi/T_0)^2>}$')
        imagename = os.path.join(self.outdir, self.data3D.outdir, "{0}".format(self.data3D.StepNumber), "enp.{0}".format(self.data3D.ext))
        fig.savefig(imagename, bbox_inches="tight")
        ax.cla()

        for i in range(1, 10):
            ax.semilogy(self.TurbTime, np.sqrt(enn[i, :]) / 16, label='n={0}'.format(i))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('$\sqrt{<|\phi_n/T_0|^2>}$')
        ax.legend()
        imagename = os.path.join(self.outdir, self.data3D.outdir, "{0}".format(self.data3d.StepNumber), "enn.{0}".format(self.data3D.ext))
        fig.savefig(imagename, bbox_inches="tight")
        ax.cla()


    class BaseData(object):

        def __init__(self, filename, ioname):
            self.filename = filename
            self.ioname = ioname
            self.on = False


        def init(self):
            self.io = adios.DeclareIO(self.ioname)
            self.engine = self.io.Open(self.filename, adios2.Mode.Read)
            self.timeout = 0.0
            self.variables = []

       
        def NewDataCheck(self):
            status = self.engine.BeginStep(adios2.StepMode.Read, self.timeout)

	    if (status == adios2.StepStatus.OK):
                NewData = True
	    elif (status == adios2.StepStatus.NotReady):
		NewData = False
	    elif (status == adios2.StepStatus.EndOfStream):
		NewData = False
                self.Stop()
            elif (status == adios2.StepStatus.OtherError):
                NewData = False
                print("1D data file {0} encountered an error in BeginStep -- closing the stream and aborting its usage", file=sys.stderr)
                self.Stop()

            return status


        def GetData(self):
            if self.variables != []:
                variables = self.variables
            else:
                variables = self.io.AvailableVariables()

            for varname in variables:
                var = self.io.InquireVariable(varname)
                shape = var.Shape()
                if len(shape) == 0:
                    shape = [1]
                else:
                    var.SetSelection([[0]*len(shape), shape])
                setattr(self, varname, np.empty(shape, dtype=kittie_common.GetType(var)))
                self.engine.Get(var, getattr(self, varname))
            self.StepNumber = engine.CurrentStep()
            self.engine.EndStep()

        def Stop(self):
            self.engine.close()
            self.on = False


