"""Module of the XGC1 loader for regerating general plots using ADIOS2
Some parts are taken from Michael's xgc.py which is taken from Loic's load_XGC_local for BES.
It reads the data from the simulation especially 1D results and other small data output.

TODO
3D data are loaded only when it is specified.
"""

import numpy as np
import os
import adios2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.tri import Triangulation
import matplotlib.ticker

import sys
import re

try:
    import kittie
except:
    kittie = None


def GetType(varid):
    size = varid.Sizeof()
    kind = varid.Type()

    # I'm just handling the common ones for now
    if kind.find('int') != -1:
        if size == 8:
            UserType = np.int64
        elif size == 4:
            UserType = np.int32
        elif size == 2:
            UserType = np.int16
    elif kind.find('double') != -1:
        UserType = np.float64
    elif (kind.find('float') != -1) or (kind.find('single') != -1):
        UserType = np.float32

    return UserType


def GetText(filename, searchpattern):
    result = None
    if os.path.exists(filename):
        with open(filename) as f:
            intxt = f.read()
        pattern = re.compile(searchpattern, re.MULTILINE)
        matches = pattern.findall(intxt)
        if len(matches) > 0:
            result = matches[-1]
    return result


class ScalarFormatterClass(matplotlib.ticker.ScalarFormatter):

   def _set_format(self):
      self.format = "%+1.2f"


class xgc1(object):


    class PlotSetup(object):

        def __init__(self, options, codename, adios):
            self.options = options
            self.gs = gridspec.GridSpec(1, 1)
            self.fig = plt.figure(tight_layout=True)
            self.ax = self.fig.add_subplot(self.gs[0, 0])

            self.codename = codename
            self.DashboardInit = False
            self.adios = adios
            self.DefaultOption('dashboard', True)

            self.DefaultOption('fontsize', 'medium')
            self.DefaultOption('ext', 'png')
            self.DefaultOption('movie', False)


        def DefaultOption(self, key, default):
            if key in self.options:
                setattr(self, key, self.options[key])
            else:
                setattr(self, key, default)


        def DashboardSave(self, plotname, step, directory):
            steparr = np.array([step])
            if not self.DashboardInit:
                ioname = plotname + ".done"
                self.DashboardIO = self.adios.DeclareIO(ioname)
                VarNumber = self.DashboardIO.DefineVariable("Step", steparr, [], [], [])
                self.DashboardEngine = self.DashboardIO.Open(ioname + ".bp", adios2.Mode.Write)
                self.DashboardInit = True

            tmp = os.path.join("{0}-images".format(plotname), "{0}".format(step))
            os.makedirs(tmp)
            tmp = os.path.join(tmp, "{0}-{1}".format(self.codename, plotname))
            os.symlink(os.path.abspath(directory), os.path.abspath(tmp))

            self.DashboardEngine.BeginStep()
            var = self.DashboardIO.InquireVariable("Step")
            self.DashboardEngine.Put(var, steparr)
            self.DashboardEngine.EndStep()



    class PlanePlot(PlotSetup):

        def __init__(self, options, xgc1, label):
            super().__init__(options, xgc1.options['codename'], xgc1.adios)
            self.DefaultOption('plane', 0)
            self.DefaultOption('cmap', 'jet')
            self.DefaultOption('levels', 50)
            self.DefaultOption('percentile', None)
            self.DefaultOption('MaxInchesX', 10)
            self.DefaultOption('MaxInchesY', 8)
            self.init = False
            xgc1.mesh.AddVariables(["rz", "nd_connect_list"])
            xgc1.dataunits.AddVariables(["sml_dt", "diag_1d_period"])
            self.label = label

    
    def __init__(self, datadir, options):
        self.datadir = datadir
        self.options = options
        self.outdir = "plots"

        adiosargs = []
        xmlfile = os.path.join(datadir, "adios2cfg.xml")
        if os.path.exists(xmlfile):
            adiosargs += [xmlfile]
        self.adios = adios2.ADIOS(*adiosargs)

        self.dataunits = self.BaseData(os.path.join(self.datadir, "xgc.units.bp"), "output.units", self.adios)
        self.f0mesh = self.BaseData(os.path.join(self.datadir, "xgc.f0.mesh.bp"), "diagnosis.f0.mesh", self.adios)
        self.mesh = self.BaseData(os.path.join(self.datadir, "xgc.mesh.bp"), "diagnosis.mesh", self.adios)
        self.data3D = self.BaseData(os.path.join(self.datadir, "xgc.3d.bp"), "field3D", self.adios, subsample_factor=options['subsample-factor-3D'], last_step=options['last-step'])

        if self.options['turbulence intensity']['use']:
            self.TurbData = self.PlotSetup(self.options['turbulence intensity'], self.options['codename'], self.adios)
            self.TurbData.DefaultOption('outdir', 'TurbulenceIntensity')
            self.TurbData.DefaultOption('psirange', [0.17, 0.4])
            self.TurbData.DefaultOption('nmodes', 9)
            self.TurbData.DefaultOption('legend', {})
            self.TurbData.Time = []

            self.data3D.AddVariables(["dpot"])
            self.dataunits.AddVariables(["sml_dt", "diag_1d_period"])
            self.f0mesh.AddVariables(["f0_T_ev"])
            self.mesh.AddVariables(["node_vol", "psi"])

        if self.options['dphi']['use']:
            self.dphi = self.PlanePlot(self.options['dphi'], self, "$\delta\phi$")
            self.dphi.DefaultOption('outdir', 'dphi')
            self.data3D.AddVariables(["dpot"])
            self.dphi.var = "dpot"
            
        if self.options['dA']['use']:
            self.options['dA']['codename'] = self.opions['codename']
            self.dA = self.PlanePlot(self.options['dA'], self, "$\delta A$")
            self.dA.DefaultOption('outdir', 'dA')
            self.data3D.AddVariables(["apars"])
            self.dA.var = "apars"


        # Start getting data

        if self.dataunits.on:
            self.dataunits.SingleRead()
            self.dataunits.psi_x = float(GetText(os.path.join(self.datadir, "units.m"), "^\s*psi_x\s*=\s*(.*)\s*;\s*$"))
            self.dataunits.diag_1d_period = int(self.dataunits.diag_1d_period)
            self.dataunits.sml_dt = float(self.dataunits.sml_dt)

        if self.f0mesh.on:
            self.f0mesh.SingleRead()

        if self.mesh.on:
            self.mesh.SingleRead()

        if self.data3D.on:
            self.data3D.init(pattern="^\s*diag_3d_period\s*=\s*(\d*).*$")
            if (self.data3D.period == None) and os.path.exists(os.path.join(self.datadir, "input")):
                self.data3D.period = self.dataunits.diag_1d_period
            if not self.data3D.perstep:
                self.data3D.AddVariables(["_StepPhysical", "_StepNumber"])


    def NotDone(self):
        return self.data3D.on


    def Close(self):
        if self.options['turbulence intensity']['use'] and self.TurbData.DashboardInit:
            self.TurbData.DashboardEngine.Close()
        if self.options['dphi']['use'] and self.dphi.DashboardInit:
            self.dphi.DashboardEngine.Close()
        if self.options['dA']['use'] and self.dA.DashboardInit:
            self.dA.DashboardEngine.Close()


    def MakePlots(self):
        new3D = False

        if self.data3D.on and self.data3D.NewDataCheck():
            self.data3D.GetData()
            new3D = True
            print("step: {0}".format(self.data3D.StepNumber)); sys.stdout.flush()

        if self.options['turbulence intensity']['use'] and new3D:
            self.TurbulenceIntensity()
        if self.options['dphi']['use'] and new3D:
            self.PlaneVarPlot(self.dphi)
        if self.options['dA']['use'] and new3D:
            self.PlaneVarPlot(self.dA)


    def PlaneVarPlot(self, PlaneObj):

        if not PlaneObj.init:
            PlaneObj.triobj = Triangulation(self.mesh.rz[:, 0], self.mesh.rz[:, 1], self.mesh.nd_connect_list)
            PlaneObj.init = True
            if PlaneObj.movie and (kittie is not None):
                PlaneObj.PlotMovie = kittie.MovieGenerator(PlaneObj.outdir)
        else:
            PlaneObj.ax = PlaneObj.fig.add_subplot(PlaneObj.gs[0, 0])

        q = getattr(self.data3D, PlaneObj.var)
        q = q[PlaneObj.plane, :] - np.mean(q, axis=0)
        if PlaneObj.percentile is not None:
            opt = np.percentile(np.fabs(q), PlaneObj.percentile)
        else:
            dpotMin = np.amin(q)
            dpotMax = np.amax(q)
            opt = np.amax(np.fabs([dpotMin, dpotMax]))
        levels = np.linspace(-opt, opt, PlaneObj.levels)
        ticks = np.linspace(-opt, opt, 7)
        
        fmt = ScalarFormatterClass(useMathText=True, useOffset=True)
        fmt.set_powerlimits((0, 1))

        ColorAxis = PlaneObj.ax.tricontourf(PlaneObj.triobj, q, cmap=PlaneObj.cmap, extend='both', levels=levels, vmin=-opt, vmax=opt)
        ColorBar = PlaneObj.fig.colorbar(ColorAxis, ax=PlaneObj.ax, pad=0, format=fmt)
        ColorBar.ax.tick_params(labelsize=PlaneObj.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(PlaneObj.fontsize)

        if self.data3D.perstep:
            time = self.dataunits.sml_dt * self.data3D.StepNumber * 1E3  # ms
        else:
            time = self.data3D._StepPhysical[0] * 1E3

        PlaneObj.ax.set_aspect(1)
        PlaneObj.ax.set_title("{1} total-f (time = {0:.3e} ms)".format(time, PlaneObj.label), fontsize=PlaneObj.fontsize)
        PlaneObj.ax.set_xlabel('r (m)', fontsize=PlaneObj.fontsize)
        PlaneObj.ax.set_ylabel('z (m)', fontsize=PlaneObj.fontsize)
        PlaneObj.ax.tick_params(axis='both', which='major', labelsize=PlaneObj.fontsize)

        xsize, ysize = PlaneObj.fig.get_size_inches()
        while True:
            xnew = xsize * 1.05
            ynew = ysize * 1.05
            if (xnew < PlaneObj.MaxInchesX) and (ynew < PlaneObj.MaxInchesY):
                xsize = xnew
                ysize = ynew
            else:
                PlaneObj.fig.set_size_inches(xsize, ysize)
                break

        outdir = os.path.join(self.outdir, PlaneObj.outdir, "{0:05d}".format(self.data3D.StepNumber))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        imagename = os.path.join(outdir, "{1}.{0}".format(PlaneObj.ext, PlaneObj.outdir))
        PlaneObj.fig.savefig(imagename, bbox_inches="tight")
        PlaneObj.fig.clear()
        if PlaneObj.movie and (kittie is not None):
            PlaneObj.PlotMovie.AddFrame(os.path.abspath(imagename))
        if PlaneObj.dashboard and (kittie is not None):
            PlaneObj.DashboardSave(PlaneObj.outdir, self.data3D.StepNumber, os.path.dirname(imagename))


    def TurbulenceIntensity(self):

        if len(self.TurbData.Time) == 0:

            psimesh = self.mesh.psi / self.dataunits.psi_x
            self.TurbData.Mask = np.nonzero( (psimesh > self.TurbData.psirange[0]) & (psimesh < self.TurbData.psirange[1]) )
            self.TurbData.Mask = self.TurbData.Mask[0]

            #self.en = np.empty( shape=(0, 0) )
            self.TurbData.enp = np.empty( shape=(0, 0) )
            self.TurbData.enn = np.empty( shape=(self.data3D.dpot.shape[0], 0))

            if self.TurbData.movie and (kittie is not None):
                self.TurbData.ennMovie = kittie.MovieGenerator('enn')
                self.TurbData.enpMovie = kittie.MovieGenerator('enp')
            
        var1 = self.data3D.dpot - np.mean(self.data3D.dpot, axis=0)
        var1 = var1 / self.f0mesh.f0_T_ev[0, :]
        varsqr = var1 * var1
        
        # Not using yet
        #s = np.mean(varsqr * self.mesh.node_vol) / np.mean(self.mesh.node_vol)
        #en = np.append(en, s)
        
        # partial sum
        sp = np.mean(varsqr[:, self.TurbData.Mask] * self.mesh.node_vol[self.TurbData.Mask]) / np.mean(self.mesh.node_vol[self.TurbData.Mask])
        self.TurbData.enp = np.append(self.TurbData.enp, sp)

        # spectral
        vft = np.abs(np.fft.fft(var1, axis=0))**2
        sn = np.mean(vft[:, self.TurbData.Mask] * self.mesh.node_vol[self.TurbData.Mask], axis=1) / np.mean(self.mesh.node_vol[self.TurbData.Mask])
        sn = sn[:, np.newaxis]
        self.TurbData.enn = np.append(self.TurbData.enn, sn, axis=1)

        if self.data3D.perstep:
            self.TurbData.Time += [self.dataunits.sml_dt * self.data3D.StepNumber * 1E3]  # ms
        else:
            self.TurbData.Time += [self.data3D._StepPhysical[0] * 1E3]
      
        outdir = os.path.join(self.outdir, self.TurbData.outdir, "{0:05d}".format(self.data3D.StepNumber))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.TurbData.ax.semilogy(self.TurbData.Time, np.sqrt(self.TurbData.enp))
        self.TurbData.ax.set_xlabel('Time (ms)', fontsize=self.TurbData.fontsize)
        self.TurbData.ax.set_ylabel('$\sqrt{<(\phi/T_0)^2>}$', fontsize=self.TurbData.fontsize)
        self.TurbData.ax.tick_params(axis='both', which='major', labelsize=self.TurbData.fontsize)
        imagename = os.path.join(outdir, "enp.{0}".format(self.TurbData.ext))
        self.TurbData.fig.savefig(imagename, bbox_inches="tight")
        self.TurbData.ax.cla()
        if self.TurbData.movie and (kittie is not None):
            self.TurbData.ennMovie.AddFrame(os.path.abspath(imagename))

        for i in range(1, self.TurbData.nmodes + 1):
            self.TurbData.ax.semilogy(self.TurbData.Time, np.sqrt(self.TurbData.enn[i, :]) / 16, label='n={0}'.format(i))
            self.TurbData.ax.set_xlabel('Time (ms)', fontsize=self.TurbData.fontsize)
            self.TurbData.ax.set_ylabel('$\sqrt{<|\phi_n/T_0|^2>}$', fontsize=self.TurbData.fontsize)
        self.TurbData.ax.legend(**self.TurbData.legend)
        imagename = os.path.join(outdir, "enn.{0}".format(self.TurbData.ext))
        self.TurbData.fig.savefig(imagename, bbox_inches="tight")
        self.TurbData.ax.cla()
        if self.TurbData.movie and (kittie is not None):
            self.TurbData.enpMovie.AddFrame(os.path.abspath(imagename))
        if self.TurbData.dashboard and (kittie is not None):
            self.TurbData.DashboardSave(self.TurbData.outdir, self.data3D.StepNumber, os.path.dirname(imagename))


    class BaseData(object):

        def __init__(self, filename, ioname, adios, perstep=None, subsample_factor=1, last_step=None):
            self.filename = filename
            self.ioname = ioname
            self.on = False
            self.variables = []
            self.adios = adios
            self.perstep = perstep
            self.subsample_factor = subsample_factor
            self.LastStep = last_step


        def AddVariables(self, variables):
            self.on = True
            for variable in variables:
                if variable not in self.variables:
                    self.variables += [variable]


        def init(self, pattern=None):
            while self.perstep is None:
                lowest = None
                matches = []
                results = os.listdir(os.path.dirname(self.filename))
                prefix = os.path.basename(self.filename)[:-2]
                for result in results:
                    if result.startswith(prefix) and result.endswith(".bp"):
                        matches += [result]

                for match in matches:
                    if (match == os.path.basename(self.filename)):
                        self.perstep = False
                        break
                    num = int(match.lstrip(prefix).rstrip(".bp"))
                    if (lowest is None) or (num < lowest):
                        lowest = num
                        self.perstep = True

            if self.perstep:
                self.filename = "{0}.{1:05d}.bp".format(self.filename[:-3], lowest)
                self.StepNumber = lowest

            self.period = None
            if pattern is not None:
                self.period = GetText(os.path.join(os.path.dirname(self.filename), "input"), pattern)
                if self.period is not None:
                    self.period = int(self.period)

            if (self.LastStep is None) and self.perstep:
                self.steps = GetText(os.path.join(os.path.dirname(self.filename), "input"), "^\s*sml_mstep\s*=\s*(\d*).*$")
                if self.steps is not None:
                    self.steps = int(self.steps)
                else:
                    print('File "input" not found, and the number of steps is ambiguous. Set "last-step" to provide one manually.', file=sys.stderr)
                    print('Quiting because there is currently no exit criteria.', file=sys.stderr)
                    sys.exit(1)

            self.io = self.adios.DeclareIO(self.ioname)
            self.engine = self.io.Open(self.filename, adios2.Mode.Read)
            self.opened = True
            self.timeout = 0.0


        def NewDataCheck(self):
            if self.perstep:

                while (self.period is None) and (not self.opened):
                    lowest = None
                    matches = []
                    results = os.listdir(os.path.dirname(self.filename))
                    prefix = os.path.basename(self.filename)[:-8]
                    for result in results:
                        if result.startswith(prefix):
                            matches += [result]
                    for match in matches:
                        if (match == os.path.basename(self.filename)):
                            continue
                        num = int(match.lstrip(prefix).rstrip(".bp"))
                        if (lowest is None) or (num < lowest):
                            lowest = num
                    if lowest is not None:
                        self.period = lowest - self.StepNumber
                        self.filename = "{0}{1:05d}.bp".format(self.filename[:-8], self.StepNumber + self.period * self.subsample_factor)

                if self.opened:
                    StepTest = self.StepNumber
                elif self.period is not None:
                    StepTest = self.StepNumber + self.period * self.subsample_factor
                if (self.LastStep is None) and (self.period is not None):
                    self.LastStep = self.StepNumber - self.period + self.steps

                if (self.period is not None) and (self.LastStep is not None) and (StepTest > self.LastStep):
                    self.on = False
                    return False
                elif not os.path.exists(self.filename):
                    return False
                elif not self.opened:
                    self.StepNumber = StepTest
                    self.engine = self.io.Open(self.filename, adios2.Mode.Read)
                    self.opened = True

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
                
            return NewData


        def SingleRead(self):
            self.init()
            while not self.NewDataCheck():
                pass
            self.GetData()
            self.Stop()


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

                if kittie is not None:
                    ADIOSType = kittie.kittie_common.GetType
                else:
                    ADIOSType = GetType
                setattr(self, varname, np.empty(shape, dtype=ADIOSType(var)))
                self.engine.Get(var, getattr(self, varname))

            """
            if not self.perstep:
                self.StepNumber = self.engine.CurrentStep()
            """

            self.engine.EndStep()

            if not self.perstep:
                self.StepNumber = getattr(self, "_StepNumber", None)
                if self.StepNumber is not None:
                    self.StepNumber = self.StepNumber[0]


            if self.perstep:
                self.engine.Close()
                self.opened = False
                if self.period is not None:
                    self.filename = "{0}{1:05d}.bp".format(self.filename[:-8], self.StepNumber + self.period * self.subsample_factor)
                self.adios.RemoveIO(self.ioname)
                self.io = self.adios.DeclareIO(self.ioname)


        def Stop(self):
            self.engine.Close()
            self.opened = False
            self.on = False


