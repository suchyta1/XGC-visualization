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
import copy
import json

from scipy.optimize import curve_fit
from scipy.special import erfc

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


def GetText(filename, searchpattern, flags=re.MULTILINE):
    result = None
    if os.path.exists(filename):
        with open(filename) as f:
            intxt = f.read()
        pattern = re.compile(searchpattern, flags)
        matches = pattern.findall(intxt)
        if len(matches) > 0:
            result = matches[-1]
    return result


def SortedIndexes(matches):
    indexes = []
    for match in matches:
        indexes += [match[-8:-3]]
    return np.sort(indexes)


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
               

        def DefineAdiosVar(self, name, data):
            var = self.io.DefineVariable(name, data, data.shape, np.zeros(len(data.shape), dtype=np.int64), data.shape)
            self.engine.Put(var, data)


        def PlotLinesADIOS(self, xlabel, ylabel, x, yarr, labelarr, xname, namearr, attrname=None, logy=False, time=0.0):
            for y, label in zip(yarr, labelarr):
                self.ax.plot(x, y, label=label)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.legend()
            imagename = os.path.join(self.outdir, "{1}-vs-{2}.{0}".format(self.ext, ylabel.replace("/","|"), xlabel).replace("/","|"))
            self.fig.savefig(imagename, bbox_inches="tight")
            self.ax.cla()
            if ('write-adios' in self.options) and self.options['write-adios']:
                """
                attr = ["lines", xname, *namearr]
                self.io.DefineAttribute(ylabel, attr)
                """
                for y, name in zip(yarr, namearr):
                    self.DefineAdiosVar(name, y)
                if xname not in self.io.AvailableVariables().keys():
                    self.DefineAdiosVar(xname, x)
                attribute = {
                        'type': 'lines',
                        'x': xname,
                        'y': namearr,
                        'yname': labelarr,
                        'xlabel': xlabel,
                        'ylabel': ylabel
                        }
                if attrname is None:
                    attrname = ylabel
                self.io.DefineAttribute(attrname, json.dumps(attribute))
                self.JsonList += [{'file_name': self.ADIOSFilename, 'attribute_name': attrname, 'group_name': self.group_name, 'time': time}]
                   

        def PlotColorADIOS(self, xlabel, ylabel, colorlabel, x, y, color, xname, yname, colorname, attrname=None, time=0.0):
            cf = self.ax.contourf(x, y, color, levels=50, cmap='jet')
            self.fig.colorbar(cf, ax=self.ax)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.set_title(colorlabel)
            imagename = os.path.join(self.outdir, "{1}-colormap.{0}".format(self.ext, colorname.replace("/","|")))
            self.fig.savefig(imagename, bbox_inches="tight")
            self.fig.clear()
            self.ax = self.fig.add_subplot(self.gs[0, 0])
            if ('write-adios' in self.options) and self.options['write-adios']:
                """
                attr = ["colormap", xname, yname, colorname]
                self.io.DefineAttribute(colorname.upper(), attr)
                """
                self.DefineAdiosVar(colorname, color)
                if xname not in self.io.AvailableVariables().keys():
                    self.DefineAdiosVar(xname, x)
                if yname not in self.io.AvailableVariables().keys():
                    self.DefineAdiosVar(yname, y)
                attribute = {
                        'type': 'colormap',
                        'x': xname,
                        'y': yname,
                        'color': colorname,
                        'xlabel': xlabel,
                        'ylabel': ylabel,
                        'title': colorlabel
                        }
                if attrname is None:
                    attrname = colorname.upper()
                self.io.DefineAttribute(attrname, json.dumps(attribute))
                self.JsonList += [{'file_name': self.ADIOSFilename, 'attribute_name': attrname, 'group_name': self.group_name, 'time': time}]


        def PlotTriColorADIOS(self, xlabel, ylabel, colorlabel, triang, color, mesh, meshname, conn, connname, colorname, attrname=None, time=0.0):
            cf = self.ax.tricontourf(triang, color, levels=self.levels, cmap=self.cmap, extend="both")
            ColorBar = self.fig.colorbar(cf, ax=self.ax)
            #ColorBar.ax.tick_params(labelsize=self.fontsize)
            #ColorBar.ax.tick_params(labelsize=self.fontsize)
            #ColorBar.ax.yaxis.offsetText.set_fontsize(self.fontsize)
            #self.ax.tick_params(axis='both', which='major', labelsize=self.fontsize)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.set_title(colorlabel)
            self.ax.set_aspect(1)
            imagename = os.path.join(self.outdir, "{1}-colormap.{0}".format(self.ext, colorname.replace("/","|")))
            self.fig.savefig(imagename, bbox_inches="tight")
            self.fig.clear()
            self.ax = self.fig.add_subplot(self.gs[0, 0])
            if ('write-adios' in self.options) and self.options['write-adios']:
                """
                attr = ["mesh-colormap", meshname, colorname]
                self.io.DefineAttribute(colorname.upper(), attr)
                """
                self.DefineAdiosVar(colorname, color)
                if meshname not in self.io.AvailableVariables().keys():
                    self.DefineAdiosVar(meshname, mesh)
                if connname not in self.io.AvailableVariables().keys():
                    self.DefineAdiosVar(connname, conn)
                attribute = {
                        'type': 'mesh-colormap',
                        'nodes': meshname,
                        'connectivity': connname,
                        'color': colorname,
                        'xlabel': xlabel,
                        'ylabel': ylabel,
                        'title': colorlabel
                        }
                if attrname is None:
                    attrname = colorname.upper()
                self.io.DefineAttribute(attrname, json.dumps(attribute))
                self.JsonList += [{'file_name': self.ADIOSFilename, 'attribute_name': attrname, 'group_name': self.group_name, 'time': time}]


        def DashboardSave(self, plotname, step):
            steparr = np.array([step])
            if not self.DashboardInit:
                ioname = plotname + ".done"
                self.DashboardIO = self.adios.DeclareIO(ioname)
                VarNumber = self.DashboardIO.DefineVariable("Step", steparr, [], [], [])
                self.DashboardEngine = self.DashboardIO.Open(ioname + ".bp", adios2.Mode.Write)
                self.DashboardInit = True
            self.DashboardEngine.BeginStep()
            var = self.DashboardIO.InquireVariable("Step")
            self.DashboardEngine.Put(var, steparr)
            self.DashboardEngine.EndStep()

        """
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
        """

    '''
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
    '''

    class PlanePlot(PlotSetup):

        def __init__(self, options, codename, adios):
            super().__init__(options, codename, adios)
            self.DefaultOption('plane', 0)
            self.DefaultOption('cmap', 'jet')
            self.DefaultOption('levels', 150)
            self.DefaultOption('percentile', None)
            self.DefaultOption('MaxInchesX', 10)
            self.DefaultOption('MaxInchesY', 8)

            self.DefaultOption('psirange', [0.8, 1.01])
            self.DefaultOption('nmodes', 8)
            self.DefaultOption('legend', {})


    def __init__(self, datadir, options):
        self.datadir = datadir
        self.options = options
        self.outdir = "plots"

        adiosargs = []
        xmlfile = os.path.join(datadir, "adios2cfg.xml")
        if os.path.exists(xmlfile):
            adiosargs += [xmlfile]
        self.adios = adios2.ADIOS(*adiosargs)

        self.dataunits = self.BaseData(os.path.join(self.datadir, "xgc.units.bp"), "output.units", self.adios, perstep=False)
        self.databfield = self.BaseData(os.path.join(self.datadir, "xgc.bfieldm.bp"), "output.bfield", self.adios, perstep=False)
        self.f0mesh = self.BaseData(os.path.join(self.datadir, "xgc.f0.mesh.bp"), "diagnosis.f0.mesh", self.adios, perstep=False)
        self.mesh = self.BaseData(os.path.join(self.datadir, "xgc.mesh.bp"), "diagnosis.mesh", self.adios, perstep=False)
        self.volumes = self.BaseData(os.path.join(self.datadir, "xgc.volumes.bp"), "output.volumes", self.adios, perstep=False)
        self.data3D = self.BaseData(os.path.join(self.datadir, "xgc.3d.bp"), "field3D", self.adios, subsample_factor=options['subsample-factor-3D'], skip=options['diag3D']['skip'])
        self.data1D = self.BaseData(os.path.join(self.datadir, "xgc.oneddiag.bp"), "diagnosis.1d", self.adios, subsample_factor=options['subsample-factor-1D'], perstep=False, skip=options['diag1D']['skip'])
        self.dataheat = self.BaseData(os.path.join(self.datadir, "xgc.heatdiag.bp"), "diagnosis.heat", self.adios, subsample_factor=options['subsample-factor-heat'], skip=options['diagheat']['skip'])

        self.dataunits.AddVariables(["sml_dt", "diag_1d_period", "sml_wedge_n", "eq_x_z", "eq_axis_r"])

        '''
        if self.options['turbulence intensity']['use']:
            self.TurbData = self.PlotSetup(self.options['turbulence intensity'], self.options['codename'], self.adios)
            self.TurbData.DefaultOption('outdir', 'TurbulenceIntensity')
            self.TurbData.DefaultOption('psirange', [0.17, 0.4])
            self.TurbData.DefaultOption('nmodes', 9)
            self.TurbData.DefaultOption('legend', {})
            self.TurbData.Time = []

            self.data3D.AddVariables(["dpot", "time"])
            self.f0mesh.AddVariables(["f0_T_ev"])
            self.mesh.AddVariables(["node_vol", "psi"])

        if self.options['dphi']['use']:
            self.dphi = self.PlanePlot(self.options['dphi'], self, "$\delta\phi$")
            self.dphi.DefaultOption('outdir', 'dphi')
            self.data3D.AddVariables(["dpot", "time"])
            self.dphi.var = "dpot"
            
        if self.options['dA']['use']:
            #self.options['dA']['codename'] = self.opions['codename']
            self.dA = self.PlanePlot(self.options['dA'], self, "$\delta A$")
            self.dA.DefaultOption('outdir', 'dA')
            self.data3D.AddVariables(["apars", "time"])
            self.dA.var = "apars"
        '''

        if self.options['diag1D']['use']:
            self.diag1D = self.PlotSetup(self.options['diag1D'], self.options['codename'], self.adios)
            self.data1D.efluxexbi = None
            self.volumes.AddVariables(["diag_1d_vol"])
            self.data1D.AddVariables([
                "time", "psi_mks", "psi", "pot00_1d", "psi00",
                "i_gc_density_df_1d", "e_gc_density_df_1d",
                "i_perp_temperature_df_1d", "e_perp_temperature_df_1d",
                "i_parallel_mean_en_df_1d", "e_parallel_mean_en_df_1d",
                "i_radial_en_flux_ExB_df_1d", "e_radial_en_flux_ExB_df_1d",
                "i_radial_en_flux_df_1d", "e_radial_en_flux_df_1d",
                "i_parallel_flow_df_1d"])
            
        if self.options['diag3D']['use']:
            self.diag3D = self.PlanePlot(self.options['diag3D'], self.options['codename'], self.adios)
            #self.diag3D.Time = []
            self.diag3D.Time = np.empty(0, dtype=np.float64)
            self.data3D.AddVariables(["dpot", "apars", "potm0", "time"])
            self.f0mesh.AddVariables(["f0_T_ev"])
            self.mesh.AddVariables(["node_vol", "psi", "rz", "nd_connect_list"])

        if self.options['diagheat']['use']:
            self.diagheat = self.PlotSetup(self.options['diagheat'], self.options['codename'], self.adios)
            self.diagheat.Time = None
            self.databfield.AddVariables(['/bfield/psi_eq_x_psi', '/bfield/rvec'])
            self.dataheat.AddVariables([
                "time",
                "e_perp_energy_psi", "i_perp_energy_psi",
                "e_para_energy_psi", "i_para_energy_psi",
                "e_number_psi", "i_number_psi",
                "psi"])


        # Start getting data

        self.dataunits.SingleRead()
        self.dataunits.psi_x = float(GetText(os.path.join(self.datadir, "units.m"), "^\s*psi_x\s*=\s*(.*)\s*;\s*$"))
        self.dataunits.ptl_ion_mass_au = float(GetText(os.path.join(self.datadir, "units.m"), "^\s*ptl_ion_mass_au\s*=\s*(.*)\s*;\s*$"))
        self.dataunits.diag_1d_period = int(self.dataunits.diag_1d_period)
        self.dataunits.sml_dt = float(self.dataunits.sml_dt)
        self.dataunits.sml_wedge_n = int(self.dataunits.sml_wedge_n)

        if self.f0mesh.on:
            self.f0mesh.SingleRead()

        if self.mesh.on:
            self.mesh.SingleRead()
            self.mesh.r = np.copy(self.mesh.rz[:, 0])
            self.mesh.z = np.copy(self.mesh.rz[:, 1])

        if self.volumes.on:
            self.volumes.SingleRead()

        if self.data1D.on:
            self.data1D.SingleRead()
            self.adios.RemoveIO(self.data1D.ioname)
            self.data1D_0 = copy.copy(self.data1D)
            self.data1D.StepCount = 0
            self.data1D.on = True
            self.data1D_0.time = self.data1D_0.time[0]
            if self.data1D_0.time != 0:
                print("First step in xgc.oneddiag.bp is not 0", file=sys.stderr)

        if self.databfield.on:
            self.databfield.SingleRead()

            self.databfield.rmid = getattr(self.databfield, '/bfield/rvec')
            """
            self.databfield.drmid = self.databfield.rmid * 0  # mem allocation
            self.databfield.drmid[1:-1] = (self.databfield.rmid[2:] - self.databfield.rmid[0:-2]) * 0.5
            self.databfield.drmid[0] = self.databfield.drmid[1]
            self.databfield.drmid[-1] = self.databfield.drmid[-2]
            """

            psi_in = getattr(self.databfield, '/bfield/psi_eq_x_psi')
            mask = np.argwhere(self.databfield.rmid > self.dataunits.eq_axis_r)
            n0 = mask[0][0]
            self.databfield.rmid0 = self.databfield.rmid[n0:]
            self.databfield.psin0 = psi_in[n0:]
            mask = np.argwhere(self.databfield.psin0 > 1)
            n0 = mask[1]
            self.databfield.dpndrs = (self.databfield.psin0[n0] - self.databfield.psin0[n0 - 1]) / (self.databfield.rmid0[n0] - self.databfield.rmid0[n0 - 1])

        if self.options['diag3D']['use']:
            self.diag3D.triobj = Triangulation(self.mesh.rz[:, 0], self.mesh.rz[:, 1], self.mesh.nd_connect_list)

        self.StartInfo = None
        self.StartTimeInit()
        self.EndTimeInit()


    def GetInput(self, pattern, flags=re.MULTILINE):
        inputfile = os.path.join(self.datadir, "input")
        if not os.path.exists(inputfile):
            print('Quiting because file "input" was not found in the data directory.', file=sys.stderr)
            sys.exit(1)
        return GetText(inputfile, pattern, flags=flags)


    def GetStartInfo(self):
        restart = self.GetInput("^\s*sml_restart\s*=\s*\.?(t|true|f|false)\.?.*$", flags=(re.MULTILINE|re.IGNORECASE))
        if (restart is None) or restart.lower().startswith('f'):
            restart = False
        elif restart.lower().startswith('t'):
            restart = True
        infofile = os.path.join(self.datadir, "runinfo.dat")

        while True:
            if not os.path.exists(infofile):
                continue
            try:
                info = np.genfromtxt(infofile, dtype=[('run_count', np.int32),('gstep', np.int32),('gstep_period1d', np.int32)])
            except:
                continue
            break
        self.StartInfo = info['gstep']
        if restart:
            try:
                self.StartInfo = self.StartInfo[-1]
            except IndexError:
                pass
        self.StartInfo = int(self.StartInfo)

   
    def StartTimeInit(self):
        if self.options['start-time'] is None:
            self.GetStartInfo()
            self.StartTime = self.StartInfo
                
        elif type(self.options['start-time']) is float:
            self.StartTime = round(self.options['start-time'] / self.dataunits.sml_dt) + 1
            
        elif type(self.options['start-time']) is int:
            self.StartTime = self.options['start-time']

        else:
            print('Invalid start time: {0}'.format(self.options['start-time']), file=sys.stderr)
            sys.exit(1)


    def EndTimeInit(self):
        if self.options['end-time'] is None:
            steps = self.GetInput("^\s*sml_mstep\s*=\s*(\d*).*$")
            if steps is None:
                print('Valid sml_mstep not found in "input"', file=sys.stderr)
                sys.exit(1)
            steps = int(steps)
            if self.StartInfo is None:
                self.GetStartInfo()
            self.EndTime = self.StartInfo - 1 + steps
            
        elif type(self.options['end-time']) is float:
            self.EndTime = round(self.options['end-time'] / self.dataunits.sml_dt) + 1

        elif type(self.options['end-time']) is int:
            self.EndTime = self.options['end-time']

        else:
            print('Invalid end time: {0}'.format(self.options['end-time']), file=sys.stderr)
            sys.exit(1)


    """
    def FindPeriod(self, data, pattern, fallback):
        data.FindPeriod(pattern=pattern)
        if data.period is None:
            data.period = fallback
    """


    def NotDone(self):
        return self.data3D.on or self.data1D.on or self.dataheat.on


    def Close(self):
        for diag in [self.diag3D, self.diag1D, self.diagheat]:
            if diag.DashboardInit:
                diag.DashboardEngine.Close()

        """
        if self.options['turbulence intensity']['use'] and self.TurbData.DashboardInit:
            self.TurbData.DashboardEngine.Close()
        if self.options['dphi']['use'] and self.dphi.DashboardInit:
            self.dphi.DashboardEngine.Close()
        if self.options['dA']['use'] and self.dA.DashboardInit:
            self.dA.DashboardEngine.Close()
        """


    def MakePlots(self):
        new3D = False
        new1D = False
        newheat = False

        if self.data3D.on and self.data3D.NewDataCheck(dt=self.dataunits.sml_dt, StartTime=self.StartTime, EndTime=self.EndTime):
            self.data3D.GetData()
            self.data3D.time = self.data3D.time[0]
            new3D = True
            print("step 3D: {0}, {1}".format(self.data3D.timestep, self.data3D.newfilename)); sys.stdout.flush()

        if self.options['diag3D']['use'] and new3D:
            self.Plot3D()

        """
        if self.options['turbulence intensity']['use'] and new3D:
            self.TurbulenceIntensity()
        if self.options['dphi']['use'] and new3D:
            self.PlaneVarPlot(self.dphi)
        if self.options['dA']['use'] and new3D:
            self.PlaneVarPlot(self.dA)
        """

        if self.data1D.on and self.data1D.NewDataCheck(dt=self.dataunits.sml_dt, StartTime=self.StartTime, EndTime=self.EndTime):
            self.data1D.GetData()
            self.data1D.time = self.data1D.time[0]
            new1D = True
            print("step 1D: {0} {1}".format(self.data1D.timestep, self.data1D.time)); sys.stdout.flush()

        if self.options['diag1D']['use'] and new1D:
            self.Plot1D()


        if self.dataheat.on and self.dataheat.NewDataCheck(dt=self.dataunits.sml_dt, StartTime=self.StartTime, EndTime=self.EndTime):
            self.dataheat.GetData()
            newheat = True
            print("step heat: {0} {1}".format(self.dataheat.timestep, self.dataheat.time)); sys.stdout.flush()

        if self.options['diagheat']['use'] and newheat:
            self.PlotHeat()


    def eich(self, xdata, q0, s, lq, dsep):
        return 0.5 * q0 * np.exp((0.5 * s / lq)**2 - (xdata - dsep) / lq) * erfc(0.5 * s / lq - (xdata - dsep) / s)

    def eich_fit1(self, ydata, rmidsepmm, pmask):
        q0init = np.max(ydata)
        sinit = 2   # 2mm
        lqinit = 1  # 1mm
        dsepinit = 0.1  # 0.1 mm
        p0 = np.array([q0init, sinit, lqinit, dsepinit])
        
        if(pmask==None):
            popt, pconv = curve_fit(self.eich, rmidsepmm, ydata, p0=p0)
        else:
            popt, pconv = curve_fit(self.eich, rmidsepmm[pmask], ydata[pmask], p0=p0)

        return popt, pconv


    '''
    def PlotHeat(self):

        # Directory setup (for dashboard)
        outdir = os.path.join("heat-images", "{0}".format(self.dataheat.timestep), "{0}-heat".format(self.diagheat.codename))
        self.diagheat.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if self.diagheat.Time is None:
            self.diagheat.Time = np.empty(0, dtype=self.dataheat.time.dtype)
            self.diagheat.dt = np.empty(0, dtype=self.dataheat.time.dtype)
            self.diagheat.qe = np.empty((2, 0), dtype=self.dataheat.e_perp_energy_psi.dtype)
            self.diagheat.qi = np.empty((2, 0), dtype=self.dataheat.i_perp_energy_psi.dtype)
            self.diagheat.e_number_psi = np.empty((2, 0), dtype=self.dataheat.e_number_psi.dtype)
            self.diagheat.i_number_psi = np.empty((2, 0), dtype=self.dataheat.i_number_psi.dtype)
            self.diagheat.lq_int = np.empty((2, 0), dtype=self.dataheat.i_perp_energy_psi.dtype)
            self.diagheat.eich_fit = np.empty((2, 0), dtype=self.dataheat.i_perp_energy_psi.dtype)

        self.diagheat.Time = np.append(self.diagheat.Time, self.dataheat.time)
        qe = self.dataunits.sml_wedge_n * (np.sum(self.dataheat.e_perp_energy_psi, axis=1) + np.sum(self.dataheat.e_para_energy_psi, axis=1))
        qi = self.dataunits.sml_wedge_n * (np.sum(self.dataheat.i_perp_energy_psi, axis=1) + np.sum(self.dataheat.i_para_energy_psi, axis=1))
        QE = self.dataheat.e_perp_energy_psi + self.dataheat.e_para_energy_psi
        QI = self.dataheat.i_perp_energy_psi + self.dataheat.i_para_energy_psi
        QT = QE + QI
        e_number_psi = np.sum(self.dataheat.e_number_psi, axis=1)
        i_number_psi = np.sum(self.dataheat.i_number_psi, axis=1)
        self.diagheat.qe = np.append(self.diagheat.qe, qe.reshape((2, 1)), axis=1)
        self.diagheat.qi = np.append(self.diagheat.qi, qi.reshape((2, 1)), axis=1)
        self.diagheat.e_number_psi = np.append(self.diagheat.e_number_psi, e_number_psi.reshape((2, 1)), axis=1)
        self.diagheat.i_number_psi = np.append(self.diagheat.i_number_psi, i_number_psi.reshape((2, 1)), axis=1)

        t0 = self.diagheat.Time * 1E3
        if ('write-adios' in self.diagheat.options) and self.diagheat.options['write-adios']:
            ioname = "diagheat.{0}".format(self.dataheat.time[0])
            filename = os.path.join(outdir, "{0}.bp".format(ioname))
            self.diagheat.io = self.adios.DeclareIO(ioname)
            self.diagheat.engine = self.diagheat.io.Open(filename, adios2.Mode.Write)
            self.diagheat.engine.BeginStep()
        
        if self.diagheat.qe.shape[1] >= 2:
            dt = self.diagheat.Time[-1] - self.diagheat.Time[-2]
            if self.diagheat.qe.shape[1] == 2:
                self.diagheat.dt = np.append(self.diagheat.dt, dt)
            self.diagheat.dt = np.append(self.diagheat.dt, dt)

            q1 = self.diagheat.qe[0] / (self.diagheat.dt * 1E6)
            q2 = self.diagheat.qi[0] / (self.diagheat.dt * 1E6)
            q3 = (self.diagheat.qi[0] + self.diagheat.qe[0]) / (self.diagheat.dt * 1E6)
            self.diagheat.ax.plot(self.diagheat.Time * 1E3, q1, label='Electron Outboard')
            self.diagheat.ax.plot(self.diagheat.Time * 1E3, q2, label='Ion Outboard')
            self.diagheat.ax.plot(self.diagheat.Time * 1E3, q3, label='Total Outboard')
            self.diagheat.ax.set_xlabel('Time (ms)')
            self.diagheat.ax.set_ylabel('Heat (MW)')
            self.diagheat.ax.legend()
            imagename = os.path.join(outdir, "Heat_Outboard.{0}".format(self.diagheat.ext))
            self.diagheat.fig.savefig(imagename, bbox_inches="tight")
            self.diagheat.ax.cla()

            q4 = self.diagheat.qe[1] / (self.diagheat.dt * 1E6)
            q5 = self.diagheat.qi[1] / (self.diagheat.dt * 1E6)
            q6 = (self.diagheat.qi[1] + self.diagheat.qe[1]) / (self.diagheat.dt * 1E6)
            self.diagheat.ax.plot(self.diagheat.Time * 1E3, q4, label='Electron Inboard')
            self.diagheat.ax.plot(self.diagheat.Time * 1E3, q5, label='Ion Inboard')
            self.diagheat.ax.plot(self.diagheat.Time * 1E3, q6, label='Total Inboard')
            self.diagheat.ax.set_xlabel('Time (ms)')
            self.diagheat.ax.set_ylabel('Heat (MW)')
            self.diagheat.ax.legend()
            imagename = os.path.join(outdir, "Heat_Inboard.{0}".format(self.diagheat.ext))
            self.diagheat.fig.savefig(imagename, bbox_inches="tight")
            self.diagheat.ax.cla()

            lq_int = np.empty((2, 1), dtype=self.dataheat.i_perp_energy_psi.dtype)
            rmidsep = np.empty(self.dataheat.i_perp_energy_psi.shape, dtype=self.dataheat.i_perp_energy_psi.dtype)
            psi_in = self.dataheat.psi / self.dataunits.psi_x
            for i in range(2):
                rmid = np.interp(psi_in[i], self.databfield.psin0, self.databfield.rmid0)
                rs = np.interp([1], psi_in[i], rmid)
                rmidsep[i] = rmid - rs
                drmid = rmid * 0  # mem allocation
                drmid[1:-1] = (rmid[2:] - rmid[0:-2]) * 0.5
                drmid[0] = drmid[1]
                drmid[-1] = drmid[-2]
                ds = (psi_in[i][1] - psi_in[i][0]) / self.databfield.dpndrs * 2 * np.pi * self.dataunits.eq_axis_r /self.dataunits.sml_wedge_n 
                QT[i] = QT[i] / dt / ds
                QE[i] = QE[i] / dt / ds
                QI[i] = QI[i] / dt / ds
                mx = np.amax(QT[i])
                lq_int[i][0] = np.sum(QT[i] * drmid) / mx

            self.diagheat.lq_int = np.append(self.diagheat.lq_int, lq_int, axis=1)
            q7 = self.diagheat.lq_int[0] * 1e3
            q8 = self.diagheat.lq_int[1] * 1e3
            self.diagheat.ax.plot(self.diagheat.Time[1:] * 1E3, q7, label='Outboard')
            self.diagheat.ax.plot(self.diagheat.Time[1:] * 1E3, q8, label='Inboard')
            self.diagheat.ax.set_xlabel('Time (ms)')
            self.diagheat.ax.set_ylabel('Lambda_q, int (mm)')
            self.diagheat.ax.legend()
            imagename = os.path.join(outdir, "Lambda_q.{0}".format(self.diagheat.ext))
            self.diagheat.fig.savefig(imagename, bbox_inches="tight")
            self.diagheat.ax.cla()
            t1 = t0[1:]

            q9  = QE[0] / 1e6
            q10 = QI[0] / 1e6
            q11 = QT[0] / 1e6
            rm_out = rmidsep[0] * 1E3
            self.diagheat.ax.plot(rmidsep[0] * 1E3, q9,  label='Electron')
            self.diagheat.ax.plot(rmidsep[0] * 1E3, q10, label='Ion')
            self.diagheat.ax.plot(rmidsep[0] * 1E3, q11, label='Total')
            self.diagheat.ax.set_xlabel('Midplane distance (mm)')
            self.diagheat.ax.set_ylabel('Heat Load (MW)')
            self.diagheat.ax.set_title('Outboard t={0:.4f} ms'.format(self.diagheat.Time[-1]*1e3))
            self.diagheat.ax.legend()
            self.diagheat.ax.set_xlim([-10, 30])
            imagename = os.path.join(outdir, "Heat_Load_Outboard_dist.{0}".format(self.diagheat.ext))
            self.diagheat.fig.savefig(imagename, bbox_inches="tight")
            self.diagheat.ax.cla()

            q12 = QE[1] / 1e6
            q13 = QI[1] / 1e6
            q14 = QT[1] / 1e6
            rm_in = rmidsep[1] * 1E3
            self.diagheat.ax.plot(rmidsep[1] * 1E3, q12, label='Electron')
            self.diagheat.ax.plot(rmidsep[1] * 1E3, q13, label='Ion')
            self.diagheat.ax.plot(rmidsep[1] * 1E3, q14, label='Total')
            self.diagheat.ax.set_xlabel('Midplane distance (mm)')
            self.diagheat.ax.set_ylabel('Heat Load (MW)')
            self.diagheat.ax.set_title('Inboard t={0:.4f} ms'.format(self.diagheat.Time[-1]*1e3))
            self.diagheat.ax.legend()
            self.diagheat.ax.set_xlim([-10, 30])
            imagename = os.path.join(outdir, "Heat_Load_Inboard_dist.{0}".format(self.diagheat.ext))
            self.diagheat.fig.savefig(imagename, bbox_inches="tight")
            self.diagheat.ax.cla()

            popt, pconv = self.eich_fit1(QT[0], rmidsep[0]*1e3, None)
            q15 = self.eich(rmidsep[0]*1e3, popt[0], popt[1], popt[2], popt[3])
            self.diagheat.ax.plot(rmidsep[0]*1e3, q15, label='Eich')
            self.diagheat.ax.plot(rmidsep[0]*1e3, QT[0], label='Heatload')
            self.diagheat.ax.set_title('t={0:.4f} ms Outboard\n $\lambda_q$={1:.3f}, S={2:.3f}'.format(self.diagheat.Time[-1]*1E3, popt[2], popt[1]))
            self.diagheat.ax.set_xlabel('Midplane distance (mm)')
            self.diagheat.ax.set_ylabel('Heat Load (W)')
            self.diagheat.ax.legend()
            self.diagheat.ax.set_xlim([-10, 30])
            imagename = os.path.join(outdir, "Outboard_eich.{0}".format(self.diagheat.ext))
            self.diagheat.fig.savefig(imagename, bbox_inches="tight")
            self.diagheat.ax.cla()
            
            eich_fit = np.empty((2, 1), dtype=self.dataheat.i_perp_energy_psi.dtype)
            popt0, pconv0 = self.eich_fit1(QT[0], rmidsep[0]*1e3, None)
            popt1, pconv1 = self.eich_fit1(QT[1], rmidsep[1]*1e3, None)
            eich_fit[0][0] = popt0[2]
            eich_fit[1][0] = popt1[2]
            self.diagheat.eich_fit = np.append(self.diagheat.eich_fit, eich_fit, axis=1)
            self.diagheat.ax.plot(self.diagheat.Time[1:] * 1E3, self.diagheat.eich_fit[0], label='Outboard')
            self.diagheat.ax.plot(self.diagheat.Time[1:] * 1E3, self.diagheat.eich_fit[1], label='Inboard')
            self.diagheat.ax.legend()
            self.diagheat.ax.set_xlabel('Time (ms)')
            self.diagheat.ax.set_ylabel('Lambda_q, eich (mm)')
            imagename = os.path.join(outdir, "lq_eich.{0}".format(self.diagheat.ext))
            self.diagheat.fig.savefig(imagename, bbox_inches="tight")
            self.diagheat.ax.cla()

            if ('write-adios' in self.diagheat.options) and self.diagheat.options['write-adios']:
                self.diagheat.DefineAdiosVar("qe_outboard", q1)
                self.diagheat.DefineAdiosVar("qi_outboard", q2)
                self.diagheat.DefineAdiosVar("qt_outboard", q3)
                self.diagheat.DefineAdiosVar("qe_inboard", q4)
                self.diagheat.DefineAdiosVar("qi_inboard", q5)
                self.diagheat.DefineAdiosVar("qt_inboard", q6)
                self.diagheat.DefineAdiosVar("lq_int_outboard", q7)
                self.diagheat.DefineAdiosVar("lq_int_inboard",  q8)
                self.diagheat.DefineAdiosVar("heatload_outboard_e", q9)
                self.diagheat.DefineAdiosVar("heatload_outboard_i", q10)
                self.diagheat.DefineAdiosVar("heatload_outboard_t", q11)
                self.diagheat.DefineAdiosVar("rmidsep_out", rm_out)
                self.diagheat.DefineAdiosVar("heatload_inboard_e",  q12)
                self.diagheat.DefineAdiosVar("heatload_inboard_i",  q13)
                self.diagheat.DefineAdiosVar("heatload_inboard_t",  q14)
                self.diagheat.DefineAdiosVar("rmidsep_in",  rm_in)
                self.diagheat.DefineAdiosVar("heatload_outboard_eich",  q15)
                self.diagheat.DefineAdiosVar("heatload_outboard_total", QT[0])
                self.diagheat.DefineAdiosVar("lq_eich_outboard", self.diagheat.eich_fit[0])
                self.diagheat.DefineAdiosVar("lq_eich_inboard",  self.diagheat.eich_fit[1])

        self.diagheat.ax.plot(self.diagheat.Time * 1E3, self.diagheat.e_number_psi[0], label='Electron Outboard')
        self.diagheat.ax.plot(self.diagheat.Time * 1E3, self.diagheat.i_number_psi[0], label='Ion Outboard')
        self.diagheat.ax.set_xlabel('Time (ms)')
        self.diagheat.ax.set_ylabel('# particle per time step')
        self.diagheat.ax.legend()
        imagename = os.path.join(outdir, "Number_Outboard.{0}".format(self.diagheat.ext))
        self.diagheat.fig.savefig(imagename, bbox_inches="tight")
        self.diagheat.ax.cla()

        self.diagheat.ax.plot(self.diagheat.Time * 1E3, self.diagheat.e_number_psi[1], label='Electron Inboard')
        self.diagheat.ax.plot(self.diagheat.Time * 1E3, self.diagheat.i_number_psi[1], label='Ion Inboard')
        self.diagheat.ax.set_xlabel('Time (ms)')
        self.diagheat.ax.set_ylabel('# particle per time step')
        self.diagheat.ax.legend()
        imagename = os.path.join(outdir, "Number_Inboard.{0}".format(self.diagheat.ext))
        self.diagheat.fig.savefig(imagename, bbox_inches="tight")
        self.diagheat.ax.cla()
         
        if ('write-adios' in self.diagheat.options) and self.diagheat.options['write-adios']:
            self.diagheat.DefineAdiosVar("ptl_e_outboard", self.diagheat.e_number_psi[0])
            self.diagheat.DefineAdiosVar("ptl_i_outboard", self.diagheat.i_number_psi[0])
            self.diagheat.DefineAdiosVar("ptl_e_inboard",  self.diagheat.e_number_psi[1])
            self.diagheat.DefineAdiosVar("ptl_i_inboard",  self.diagheat.i_number_psi[1])
            self.diagheat.DefineAdiosVar("time", t0)
            self.diagheat.engine.EndStep()
            self.diagheat.engine.Close()

        self.diagheat.DashboardSave("heat", self.data3D.timestep)
    '''


    def PlotHeat(self):

        # Directory setup (for dashboard)
        outdir = os.path.join("heat-images", "{0}".format(self.dataheat.timestep), "{0}-heat".format(self.diagheat.codename))
        self.diagheat.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if self.diagheat.Time is None:
            self.diagheat.Time = np.empty(0, dtype=self.dataheat.time.dtype)
            self.diagheat.dt = np.empty(0, dtype=self.dataheat.time.dtype)
            self.diagheat.qe = np.empty((2, 0), dtype=self.dataheat.e_perp_energy_psi.dtype)
            self.diagheat.qi = np.empty((2, 0), dtype=self.dataheat.i_perp_energy_psi.dtype)
            self.diagheat.e_number_psi = np.empty((2, 0), dtype=self.dataheat.e_number_psi.dtype)
            self.diagheat.i_number_psi = np.empty((2, 0), dtype=self.dataheat.i_number_psi.dtype)
            self.diagheat.lq_int = np.empty((2, 0), dtype=self.dataheat.i_perp_energy_psi.dtype)
            self.diagheat.eich_fit = np.empty((2, 0), dtype=self.dataheat.i_perp_energy_psi.dtype)

        self.diagheat.Time = np.append(self.diagheat.Time, self.dataheat.time)
        qe = self.dataunits.sml_wedge_n * (np.sum(self.dataheat.e_perp_energy_psi, axis=1) + np.sum(self.dataheat.e_para_energy_psi, axis=1))
        qi = self.dataunits.sml_wedge_n * (np.sum(self.dataheat.i_perp_energy_psi, axis=1) + np.sum(self.dataheat.i_para_energy_psi, axis=1))
        QE = self.dataheat.e_perp_energy_psi + self.dataheat.e_para_energy_psi
        QI = self.dataheat.i_perp_energy_psi + self.dataheat.i_para_energy_psi
        QT = QE + QI
        e_number_psi = np.sum(self.dataheat.e_number_psi, axis=1)
        i_number_psi = np.sum(self.dataheat.i_number_psi, axis=1)
        self.diagheat.qe = np.append(self.diagheat.qe, qe.reshape((2, 1)), axis=1)
        self.diagheat.qi = np.append(self.diagheat.qi, qi.reshape((2, 1)), axis=1)
        self.diagheat.e_number_psi = np.append(self.diagheat.e_number_psi, e_number_psi.reshape((2, 1)), axis=1)
        self.diagheat.i_number_psi = np.append(self.diagheat.i_number_psi, i_number_psi.reshape((2, 1)), axis=1)

        t0 = self.diagheat.Time * 1E3
        if ('write-adios' in self.diagheat.options) and self.diagheat.options['write-adios']:
            self.diagheat.group_name = "diag-heat"
            ioname = "{1}.{0:.6f}".format(self.dataheat.time[0]*1e3, self.diagheat.group_name)
            self.diagheat.ADIOSFilename = os.path.join(outdir, "{0}.bp".format(ioname))
            self.diagheat.io = self.adios.DeclareIO(ioname)
            self.diagheat.engine = self.diagheat.io.Open(self.diagheat.ADIOSFilename, adios2.Mode.Write)
            self.diagheat.engine.BeginStep()
            self.diagheat.JsonList = []
        
        if self.diagheat.qe.shape[1] >= 2:
            dt = self.diagheat.Time[-1] - self.diagheat.Time[-2]
            if self.diagheat.qe.shape[1] == 2:
                self.diagheat.dt = np.append(self.diagheat.dt, dt)
            self.diagheat.dt = np.append(self.diagheat.dt, dt)

            q1 = self.diagheat.qe[0] / (self.diagheat.dt * 1E6)
            q2 = self.diagheat.qi[0] / (self.diagheat.dt * 1E6)
            q3 = (self.diagheat.qi[0] + self.diagheat.qe[0]) / (self.diagheat.dt * 1E6)
            self.diagheat.PlotLinesADIOS('Time (ms)', 'Heat (MW) [outboard]', t0, [q1,q2,q3], ['Electron Outboard', 'Ion Outboard', 'Total Outboard'], "time", ["qe_outboard", "qi_outboard", "qt_outboard"], time=self.dataheat.time[0])

            q4 = self.diagheat.qe[1] / (self.diagheat.dt * 1E6)
            q5 = self.diagheat.qi[1] / (self.diagheat.dt * 1E6)
            q6 = (self.diagheat.qi[1] + self.diagheat.qe[1]) / (self.diagheat.dt * 1E6)
            self.diagheat.PlotLinesADIOS('Time (ms)', 'Heat (MW) [inboard]', t0, [q4,q5,q6], ['Electron Inboard', 'Ion Inboard', 'Total Inboard'], "time", ["qe_inboard", "qi_inboard", "qt_inboard"], time=self.dataheat.time[0])

            lq_int = np.empty((2, 1), dtype=self.dataheat.i_perp_energy_psi.dtype)
            rmidsep = np.empty(self.dataheat.i_perp_energy_psi.shape, dtype=self.dataheat.i_perp_energy_psi.dtype)
            psi_in = self.dataheat.psi / self.dataunits.psi_x
            for i in range(2):
                rmid = np.interp(psi_in[i], self.databfield.psin0, self.databfield.rmid0)
                rs = np.interp([1], psi_in[i], rmid)
                rmidsep[i] = rmid - rs
                drmid = rmid * 0  # mem allocation
                drmid[1:-1] = (rmid[2:] - rmid[0:-2]) * 0.5
                drmid[0] = drmid[1]
                drmid[-1] = drmid[-2]
                ds = (psi_in[i][1] - psi_in[i][0]) / self.databfield.dpndrs * 2 * np.pi * self.dataunits.eq_axis_r /self.dataunits.sml_wedge_n 
                QT[i] = QT[i] / dt / ds
                QE[i] = QE[i] / dt / ds
                QI[i] = QI[i] / dt / ds
                mx = np.amax(QT[i])
                lq_int[i][0] = np.sum(QT[i] * drmid) / mx

            self.diagheat.lq_int = np.append(self.diagheat.lq_int, lq_int, axis=1)
            q7 = self.diagheat.lq_int[0] * 1e3
            q8 = self.diagheat.lq_int[1] * 1e3
            t1 = t0[1:]
            self.diagheat.PlotLinesADIOS('Time (ms)', 'Lambda_q, int (mm)',  t1, [q7,q8], ['Outboard', 'Inboard'],  "Time", ["lq_int_outboard", "lq_int_inboard"], time=self.dataheat.time[0])

            q9  = QE[0] / 1e6
            q10 = QI[0] / 1e6
            q11 = QT[0] / 1e6
            rm_out = rmidsep[0] * 1E3
            self.diagheat.PlotLinesADIOS('Midplane distance (mm)', 'Heat Load (MW) [outboard, vs. dist]', rm_out, [q9,q10,q11], ['Electron','Ion','Total'],  "rmidsep_out", ["heatload_outboard_e","heatload_outboard_i","heatload_outboard_t"], time=self.dataheat.time[0])

            q12 = QE[1] / 1e6
            q13 = QI[1] / 1e6
            q14 = QT[1] / 1e6
            rm_in = rmidsep[1] * 1E3
            self.diagheat.PlotLinesADIOS('Midplane distance (mm)', 'Heat Load (MW) [inboard, vs. dist]', rm_in, [q12,q13,q14], ['Electron','Ion','Total'],  "rmidsep_in", ["heatload_inboard_e","heatload_inboard_i","heatload_inboard_t"], time=self.dataheat.time[0])

            popt, pconv = self.eich_fit1(QT[0], rmidsep[0]*1e3, None)
            q15 = self.eich(rmidsep[0]*1e3, popt[0], popt[1], popt[2], popt[3])
            self.diagheat.PlotLinesADIOS('Midplane distance (mm)', 'Heat Load (MW) [outboard, eich]', rm_out, [q15,QT[0]], ['Eich','Heatload'], "rmidsep_out", ["heatload_outboard_eich", "heatload_outboard_total"], time=self.dataheat.time[0])
            
            eich_fit = np.empty((2, 1), dtype=self.dataheat.i_perp_energy_psi.dtype)
            popt0, pconv0 = self.eich_fit1(QT[0], rmidsep[0]*1e3, None)
            popt1, pconv1 = self.eich_fit1(QT[1], rmidsep[1]*1e3, None)
            eich_fit[0][0] = popt0[2]
            eich_fit[1][0] = popt1[2]
            self.diagheat.eich_fit = np.append(self.diagheat.eich_fit, eich_fit, axis=1)
            self.diagheat.PlotLinesADIOS('Time (ms)', 'Lambda_q, eich (mm)', t1, [self.diagheat.eich_fit[0], self.diagheat.eich_fit[1]], ['Outboard', 'Inboard'],  "Time", ["lq_eich_outboard", "lq_eich_inboard"], time=self.dataheat.time[0])

        self.diagheat.PlotLinesADIOS('Time (ms)', '# particle per time step [outboard]', t0, [self.diagheat.e_number_psi[0], self.diagheat.i_number_psi[0]], ['Electron Outboard', 'Ion Outboard'], "time", ["ptl_e_outboard", "ptl_i_outboard"], time=self.dataheat.time[0])
        self.diagheat.PlotLinesADIOS('Time (ms)', '# particle per time step [inboard]',  t0, [self.diagheat.e_number_psi[1], self.diagheat.i_number_psi[1]], ['Electron Inboard',  'Ion Inboard'],  "time", ["ptl_e_inboard",  "ptl_i_inboard"], time=self.dataheat.time[0])
        
        if ('write-adios' in self.diagheat.options) and self.diagheat.options['write-adios']:
            self.diagheat.engine.EndStep()
            self.diagheat.engine.Close()
            JsonFilename = os.path.join(outdir, "plots.json")
            with open(JsonFilename, "w") as outfile:
                outfile.write(json.dumps(self.diagheat.JsonList, indent=4))

        self.diagheat.DashboardSave("heat", self.dataheat.timestep)


    '''
    def Plot1D(self):

        # Directory setup (for dashboard)
        outdir = os.path.join("1D-images", "{0}".format(self.data1D.timestep), "{0}-1D".format(self.diag1D.codename))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Electron density
        self.diag1D.ax.plot(self.data1D_0.psi, self.data1D_0.e_gc_density_df_1d, label="Initial")
        self.diag1D.ax.plot(self.data1D.psi,   self.data1D.e_gc_density_df_1d,   label="t={0:.4f} ms".format(self.data1D.time * 1e3))
        self.diag1D.ax.set(xlabel='Normalized Pol. Flux')
        self.diag1D.ax.set(ylabel='Elec. g.c. Density (m^-3)')
        self.diag1D.ax.legend()
        imagename = os.path.join(outdir, "Electron_gc_Density.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()
        
        # Ion density
        self.diag1D.ax.plot(self.data1D_0.psi, self.data1D_0.i_gc_density_df_1d, label="Initial")
        self.diag1D.ax.plot(self.data1D.psi,   self.data1D.i_gc_density_df_1d,   label="t={0:.4f} ms".format(self.data1D.time * 1e3))
        self.diag1D.ax.set(xlabel='Normalized Pol. Flux')
        self.diag1D.ax.set(ylabel='Ion g.c. Density (m^-3)')
        self.diag1D.ax.legend()
        imagename = os.path.join(outdir, "Ion_gc_Density.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()

        # Electron temperature
        Te = (self.data1D.e_perp_temperature_df_1d + self.data1D.e_parallel_mean_en_df_1d)/3*2
        Te_0 = (self.data1D_0.e_perp_temperature_df_1d + self.data1D_0.e_parallel_mean_en_df_1d)/3*2
        self.diag1D.ax.plot(self.data1D_0.psi, Te_0, label="Initial")
        self.diag1D.ax.plot(self.data1D.psi,   Te,   label="t={0:.4f} ms".format(self.data1D.time * 1e3))
        self.diag1D.ax.set(xlabel='Normalized Pol. Flux')
        self.diag1D.ax.set(ylabel='Elec. Temperature (eV)')
        self.diag1D.ax.legend()
        imagename = os.path.join(outdir, "Electron_temperature.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()

        # Ion temperature
        Ti = (self.data1D.i_perp_temperature_df_1d + self.data1D.i_parallel_mean_en_df_1d)/3*2
        Ti_0 = (self.data1D_0.i_perp_temperature_df_1d + self.data1D_0.i_parallel_mean_en_df_1d)/3*2
        self.diag1D.ax.plot(self.data1D_0.psi, Ti_0, label="Initial")
        self.diag1D.ax.plot(self.data1D.psi,   Ti,   label="t={0:.4f} ms".format(self.data1D.time * 1e3))
        self.diag1D.ax.set(xlabel='Normalized Pol. Flux')
        self.diag1D.ax.set(ylabel='Ion Temperature (eV)')
        self.diag1D.ax.legend()
        imagename = os.path.join(outdir, "Ion_temperature.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()

        # Ion parallel flow
        self.diag1D.ax.plot(self.data1D_0.psi, self.data1D_0.i_parallel_flow_df_1d, label="Initial")
        self.diag1D.ax.plot(self.data1D.psi,   self.data1D.i_parallel_flow_df_1d,   label="t={0:.4f} ms".format(self.data1D.time * 1e3))
        self.diag1D.ax.set(xlabel='Normalized Pol. Flux')
        self.diag1D.ax.set(ylabel='Ion. parallel flow (m/s)')
        self.diag1D.ax.legend()
        imagename = os.path.join(outdir, "Ion_parallel_flow.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()

        # Potential
        p0 = self.data1D_0.psi00/self.dataunits.psi_x
        p1 = self.data1D.psi00/self.dataunits.psi_x
        self.diag1D.ax.plot(p0, self.data1D_0.pot00_1d, label="Initial")
        self.diag1D.ax.plot(p1,   self.data1D.pot00_1d,   label="t={0:.4f} ms".format(self.data1D.time * 1e3))
        self.diag1D.ax.set(xlabel='psi00/psix')
        self.diag1D.ax.set(ylabel='Potential (V)')
        self.diag1D.ax.legend()
        imagename = os.path.join(outdir, "potential.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()

        # Calculation derivatives
        dpsi = np.zeros_like(self.data1D.psi_mks)
        dpsi[1:-1] = 0.5 * (self.data1D.psi_mks[2:] - self.data1D.psi_mks[0:-2])
        dpsi[0] = dpsi[1]
        dpsi[-1] = dpsi[-2]
        dvdp = self.volumes.diag_1d_vol / dpsi
        dvdpall = dvdp * self.dataunits.sml_wedge_n

        # Heat Fluxes ExB
        efluxexbi = self.data1D.i_gc_density_df_1d * self.data1D.i_radial_en_flux_ExB_df_1d * dvdpall
        efluxexbe = self.data1D.e_gc_density_df_1d * self.data1D.e_radial_en_flux_ExB_df_1d * dvdpall
        f1 = efluxexbi/1E6
        f2 = efluxexbe/1E6
        self.diag1D.ax.plot(self.data1D.psi, f1, label='Ion flux')
        self.diag1D.ax.plot(self.data1D.psi, f2, label='Elec. flux')
        self.diag1D.ax.legend()
        self.diag1D.ax.set(xlabel='Normalized Pol. Flux')
        self.diag1D.ax.set(ylabel='Radial Heat Flux (MW)')
        self.diag1D.ax.set(title='Radial Heat Flux by ExB at t={0:.4f} ms'.format(self.data1D.time * 1e3))
        imagename = os.path.join(outdir, "Radial_Heat_Flux_ExB.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()

        # Heat Fluxes
        efluxi = self.data1D.i_gc_density_df_1d * self.data1D.i_radial_en_flux_df_1d * dvdpall
        efluxe = self.data1D.e_gc_density_df_1d * self.data1D.e_radial_en_flux_df_1d * dvdpall
        f3 = efluxi/1E6
        f4 = efluxe/1E6
        self.diag1D.ax.plot(self.data1D.psi, f3, label='Ion flux')
        self.diag1D.ax.plot(self.data1D.psi, f4, label='Elec. flux')
        self.diag1D.ax.legend()
        self.diag1D.ax.set(xlabel='Normalized Pol. Flux')
        self.diag1D.ax.set(ylabel='Radial Heat Flux (MW)')
        self.diag1D.ax.set(title='Radial Heat Flux at t={0:.4f} ms'.format(self.data1D.time * 1e3))
        imagename = os.path.join(outdir, "Radial_Heat_Flux.{0}".format(self.diag1D.ext))
        self.diag1D.fig.savefig(imagename, bbox_inches="tight")
        self.diag1D.ax.cla()

        # Save Fluxes for 2D color map
        if self.data1D.efluxexbi is None:
            self.data1D.efluxexbi = np.empty((0, efluxexbi.shape[0]), dtype=efluxexbi.dtype)
            self.data1D.efluxi = np.empty((0, efluxi.shape[0]), dtype=efluxi.dtype)
            self.data1D.timeflux = np.empty(0, dtype=np.float64)
        self.data1D.efluxexbi = np.append(self.data1D.efluxexbi, efluxexbi.reshape((1, efluxexbi.shape[0])), axis=0)
        self.data1D.efluxi = np.append(self.data1D.efluxi, efluxi.reshape((1, efluxi.shape[0])), axis=0)
        self.data1D.timeflux = np.append(self.data1D.timeflux, self.data1D.time)

        if ('write-adios' in self.diag1D.options) and self.diag1D.options['write-adios']:
            ioname = "diag1D.{0}".format(self.data1D.time)
            filename = os.path.join(outdir, "{0}.bp".format(ioname))
            self.diag1D.io = self.adios.DeclareIO(ioname)
            self.diag1D.engine = self.diag1D.io.Open(filename, adios2.Mode.Write)
            self.diag1D.engine.BeginStep()

        # 2D Heat Flux ExB
        if self.data1D.timeflux.shape[0] > 1:
            cf = self.diag1D.ax.contourf(self.data1D.psi, self.data1D.timeflux*1E3, self.data1D.efluxexbi/1E6, levels=50, cmap='jet')
            self.diag1D.fig.colorbar(cf, ax=self.diag1D.ax)
            self.diag1D.ax.set_title('Ion Heat Flux by ExB (MW)')
            self.diag1D.ax.set_xlabel('Poloidal Flux')
            self.diag1D.ax.set_ylabel('Time (ms)')
            imagename = os.path.join(outdir, "Ion_Heat_Flux_ExB_2D.{0}".format(self.diag1D.ext))
            self.diag1D.fig.savefig(imagename, bbox_inches="tight")
            self.diag1D.fig.clear()
            self.diag1D.ax = self.diag1D.fig.add_subplot(self.diag1D.gs[0, 0])
            if ('write-adios' in self.diag1D.options) and self.diag1D.options['write-adios']:
                t1 = self.data1D.timeflux*1E3
                t2 = self.data1D.efluxexbi/1E6
                self.diag1D.DefineAdiosVar("timeflux", t1)
                self.diag1D.DefineAdiosVar("i_flux_r_ExB_2D", t2)

        # 2D Heat Flux
        if self.data1D.timeflux.shape[0] > 1:
            cf = self.diag1D.ax.contourf(self.data1D.psi, self.data1D.timeflux*1E3, self.data1D.efluxi/1E6, levels=50, cmap='jet')
            self.diag1D.fig.colorbar(cf, ax=self.diag1D.ax)
            self.diag1D.ax.set_title('Ion Heat Flux (MW)')
            self.diag1D.ax.set_xlabel('Poloidal Flux')
            self.diag1D.ax.set_ylabel('Time (ms)')
            imagename = os.path.join(outdir, "Ion_Heat_Flux_2D.{0}".format(self.diag1D.ext))
            self.diag1D.fig.savefig(imagename, bbox_inches="tight")
            self.diag1D.fig.clear()
            self.diag1D.ax = self.diag1D.fig.add_subplot(self.diag1D.gs[0, 0])
            if ('write-adios' in self.diag1D.options) and self.diag1D.options['write-adios']:
                t3 = self.data1D.efluxi/1E6
                self.diag1D.DefineAdiosVar("i_flux_r_2D", t3)

        if ('write-adios' in self.diag1D.options) and self.diag1D.options['write-adios']:
            self.diag1D.DefineAdiosVar("psi_0", self.data1D_0.psi)
            self.diag1D.DefineAdiosVar("psi", self.data1D.psi)
            self.diag1D.DefineAdiosVar("psi00_0/psi_x", p0)
            self.diag1D.DefineAdiosVar("psi00/psi_x", p1)
            self.diag1D.DefineAdiosVar("e_gc_density_df_1d_0", self.data1D_0.e_gc_density_df_1d)
            self.diag1D.DefineAdiosVar("e_gc_density_df_1d",   self.data1D.e_gc_density_df_1d)
            self.diag1D.DefineAdiosVar("i_gc_density_df_1d_0", self.data1D_0.i_gc_density_df_1d)
            self.diag1D.DefineAdiosVar("i_gc_density_df_1d",   self.data1D.i_gc_density_df_1d)
            self.diag1D.DefineAdiosVar("Te_0", Te_0)
            self.diag1D.DefineAdiosVar("Te", Te)
            self.diag1D.DefineAdiosVar("Ti_0", Ti_0)
            self.diag1D.DefineAdiosVar("Ti", Ti)
            self.diag1D.DefineAdiosVar("i_parallel_flow_df_1d_0", self.data1D_0.i_parallel_flow_df_1d)
            self.diag1D.DefineAdiosVar("i_parallel_flow_df_1d", self.data1D.i_parallel_flow_df_1d)
            self.diag1D.DefineAdiosVar("self.data1D_0.pot00_1d_0", self.data1D_0.pot00_1d)
            self.diag1D.DefineAdiosVar("self.data1D_0.pot00_1d", self.data1D.pot00_1d)
            self.diag1D.DefineAdiosVar("i_flux_r_ExB", f1)
            self.diag1D.DefineAdiosVar("e_flux_r_ExB", f2)
            self.diag1D.DefineAdiosVar("i_flux_r", f3)
            self.diag1D.DefineAdiosVar("e_flux_r", f4)
            self.diag1D.engine.EndStep()
            self.diag1D.engine.Close()

        self.diag1D.DashboardSave("1D", self.data3D.timestep)
    '''

    def Plot1D(self):

        # Directory setup (for dashboard)
        outdir = os.path.join("1D-images", "{0}".format(self.data1D.timestep), "{0}-1D".format(self.diag1D.codename))
        self.diag1D.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if ('write-adios' in self.diag1D.options) and self.diag1D.options['write-adios']:
            self.diag1D.group_name = "diag-1D"
            ioname = "{1}.{0:.6f}".format(self.data1D.time*1e3, self.diag1D.group_name)
            self.diag1D.ADIOSFilename = os.path.join(outdir, "{0}.bp".format(ioname))
            self.diag1D.io = self.adios.DeclareIO(ioname)
            self.diag1D.engine = self.diag1D.io.Open(self.diag1D.ADIOSFilename, adios2.Mode.Write)
            self.diag1D.engine.BeginStep()
            self.diag1D.JsonList = []

        # Electron density
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Elec. g.c. Density (m^-3)', self.data1D.psi, [self.data1D_0.e_gc_density_df_1d,self.data1D.e_gc_density_df_1d], ['Initial',"t={0:.4f} ms".format(self.data1D.time * 1e3)], "psi", ["e_gc_density_df_1d_0","e_gc_density_df_1d"], time=self.data1D.time)
 
        # Ion density
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Ion g.c. Density (m^-3)', self.data1D.psi, [self.data1D_0.i_gc_density_df_1d,self.data1D.i_gc_density_df_1d], ['Initial',"t={0:.4f} ms".format(self.data1D.time * 1e3)], "psi", ["i_gc_density_df_1d_0","i_gc_density_df_1d"], time=self.data1D.time)

        # Electron temperature
        Te = (self.data1D.e_perp_temperature_df_1d + self.data1D.e_parallel_mean_en_df_1d)/3*2
        Te_0 = (self.data1D_0.e_perp_temperature_df_1d + self.data1D_0.e_parallel_mean_en_df_1d)/3*2
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Elec. Temperature (eV)', self.data1D.psi, [Te_0,Te], ['Initial',"t={0:.4f} ms".format(self.data1D.time * 1e3)], "psi", ["Te_0","Te"], time=self.data1D.time)

        # Ion temperature
        Ti = (self.data1D.i_perp_temperature_df_1d + self.data1D.i_parallel_mean_en_df_1d)/3*2
        Ti_0 = (self.data1D_0.i_perp_temperature_df_1d + self.data1D_0.i_parallel_mean_en_df_1d)/3*2
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Ion Temperature (eV)', self.data1D.psi, [Ti_0,Ti], ['Initial',"t={0:.4f} ms".format(self.data1D.time * 1e3)], "psi", ["Ti_0","Ti"], time=self.data1D.time)

        # Ion parallel flow
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Ion. parallel flow (m/s)', self.data1D.psi, [self.data1D_0.i_parallel_flow_df_1d, self.data1D.i_parallel_flow_df_1d], ['Initial',"t={0:.4f} ms".format(self.data1D.time * 1e3)], "psi", ["i_parallel_flow_df_1d_0","i_parallel_flow_df_1d"], time=self.data1D.time)

        # Potential
        p0 = self.data1D_0.psi00/self.dataunits.psi_x
        p1 = self.data1D.psi00/self.dataunits.psi_x
        self.diag1D.PlotLinesADIOS('psi00/psix', 'Potential (V)', p1, [self.data1D_0.pot00_1d, self.data1D.pot00_1d], ['Initial',"t={0:.4f} ms".format(self.data1D.time * 1e3)], "psi00/psi_x", ["pot00_1d_0", "pot00_1d"], time=self.data1D.time)

        # Calculation derivatives
        dpsi = np.zeros_like(self.data1D.psi_mks)
        dpsi[1:-1] = 0.5 * (self.data1D.psi_mks[2:] - self.data1D.psi_mks[0:-2])
        dpsi[0] = dpsi[1]
        dpsi[-1] = dpsi[-2]
        dvdp = self.volumes.diag_1d_vol / dpsi
        dvdpall = dvdp * self.dataunits.sml_wedge_n

        # Heat Fluxes ExB
        efluxexbi = self.data1D.i_gc_density_df_1d * self.data1D.i_radial_en_flux_ExB_df_1d * dvdpall
        efluxexbe = self.data1D.e_gc_density_df_1d * self.data1D.e_radial_en_flux_ExB_df_1d * dvdpall
        f1 = efluxexbi/1E6
        f2 = efluxexbe/1E6
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Radial Heat Flux by ExB (MW)', self.data1D.psi, [f1,f2], ['Ion flux', 'Elec. flux'], "psi", ["i_flux_r_ExB","e_flux_r_ExB"], time=self.data1D.time)

        # Heat Fluxes
        efluxi = self.data1D.i_gc_density_df_1d * self.data1D.i_radial_en_flux_df_1d * dvdpall
        efluxe = self.data1D.e_gc_density_df_1d * self.data1D.e_radial_en_flux_df_1d * dvdpall
        f3 = efluxi/1E6
        f4 = efluxe/1E6
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Radial Heat Flux (MW)', self.data1D.psi, [f3,f4], ['Ion flux', 'Elec. flux'], "psi", ["i_flux_r","e_flux_r"], time=self.data1D.time)

        # Save Fluxes for 2D color map
        if self.data1D.efluxexbi is None:
            self.data1D.efluxexbi = np.empty((0, efluxexbi.shape[0]), dtype=efluxexbi.dtype)
            self.data1D.efluxi = np.empty((0, efluxi.shape[0]), dtype=efluxi.dtype)
            self.data1D.timeflux = np.empty(0, dtype=np.float64)
        self.data1D.efluxexbi = np.append(self.data1D.efluxexbi, efluxexbi.reshape((1, efluxexbi.shape[0])), axis=0)
        self.data1D.efluxi = np.append(self.data1D.efluxi, efluxi.reshape((1, efluxi.shape[0])), axis=0)
        self.data1D.timeflux = np.append(self.data1D.timeflux, self.data1D.time)

        # 2D Heat Flux ExB
        if self.data1D.timeflux.shape[0] > 1:
            t1 = self.data1D.timeflux*1E3
            c1 = self.data1D.efluxexbi/1E6
            self.diag1D.PlotColorADIOS('Normalized Pol. Flux', 'Time (ms)', 'Radial Heat Flux by ExB (MW)', self.data1D.psi, t1, c1, "psi", "timeflux", "i_flux_r_ExB_2D", time=self.data1D.time)
            #self.diag1D.PlotColorADIOS('Normalized Pol. Flux', 'Time (ms)', 'Radial Heat Flux by ExB (MW)', self.data1D.psi, t1, c1, "psi", "timeflux", "i_flux_r_ExB_2D", attrname="Radial Heat Flux by ExB (MW) [2D]", time=self.data1D.time)

        # 2D Heat Flux
        if self.data1D.timeflux.shape[0] > 1:
            t2 = self.data1D.timeflux*1E3
            c2 = self.data1D.efluxi/1E6
            self.diag1D.PlotColorADIOS('Normalized Pol. Flux', 'Time (ms)', 'Radial Heat Flux (MW)', self.data1D.psi, t2, c2, "psi", "timeflux", "i_flux_r_2D", time=self.data1D.time)
            #self.diag1D.PlotColorADIOS('Normalized Pol. Flux', 'Time (ms)', 'Radial Heat Flux (MW)', self.data1D.psi, t2, c2, "psi", "timeflux", "i_flux_r_2D", attrname="Radial Heat Flux (MW) [2D]", time=self.data1D.time)

        if ('write-adios' in self.diag1D.options) and self.diag1D.options['write-adios']:
            self.diag1D.engine.EndStep()
            self.diag1D.engine.Close()
            JsonFilename = os.path.join(outdir, "plots.json")
            with open(JsonFilename, "w") as outfile:
                outfile.write(json.dumps(self.diag1D.JsonList, indent=4))

        self.diag1D.DashboardSave("1D", self.data1D.timestep)


    """
    def Plot3D(self):

        # Directory setup (for dashboard)
        outdir = os.path.join("mesh-images", "{0}".format(self.data3D.timestep), "{0}-mesh".format(self.diag3D.codename))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Normalized dpot
        q_dpot = (self.data3D.dpot[self.diag3D.plane, :] - np.mean(self.data3D.dpot, axis=0)) / self.f0mesh.f0_T_ev[0, :]
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_dpot, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$\Phi/T_e$ w/o n=0 at %2.4f ms' % (self.data3D.time * 1e3) 
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "dpot.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])

        # pot n=0 mode
        q_potm0 = self.data3D.potm0
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_potm0, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$\Phi$ n=0 at %2.4f ms' % (self.data3D.time * 1e3)
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "potm0.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])

        # Normalized Apar
        imass = self.dataunits.ptl_ion_mass_au * 1.67E-27
        vth_f0 = np.sqrt(1.6E-19 * self.f0mesh.f0_T_ev[0, :] / imass)
        q_apars = self.data3D.apars[self.diag3D.plane, :] * vth_f0 / self.f0mesh.f0_T_ev[0, :]
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_apars, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$A_{||} C_s/T_e$ at %2.4f ms' % (self.data3D.time * 1e3) 
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "apars.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])

        if ('write-adios' in self.diag3D.options) and self.diag3D.options['write-adios']:
            ioname = "diag3D.{0}".format(self.data3D.time)
            filename = os.path.join(outdir, "{0}.bp".format(ioname))
            io = self.adios.DeclareIO(ioname)
            engine = io.Open(filename, adios2.Mode.Write)
            var_dpot  = io.DefineVariable("q_dpot",  q_dpot,  q_dpot.shape,  [0], q_dpot.shape)
            var_potm0 = io.DefineVariable("q_potm0", q_potm0, q_potm0.shape, [0], q_potm0.shape)
            var_apars = io.DefineVariable("q_apars", q_apars, q_apars.shape, [0], q_apars.shape)
            var_rz = io.DefineVariable("mesh_rz", self.mesh.rz, self.mesh.rz.shape, [0,0], self.mesh.rz.shape)
            var_r = io.DefineVariable("mesh_r", self.mesh.r, self.mesh.r.shape, [0], self.mesh.r.shape)
            var_z = io.DefineVariable("mesh_z", self.mesh.z, self.mesh.z.shape, [0], self.mesh.z.shape)
            var_c = io.DefineVariable("mesh_connectivity", self.mesh.nd_connect_list, self.mesh.nd_connect_list.shape, [0, 0], self.mesh.nd_connect_list.shape)
            engine.BeginStep()
            engine.Put(var_dpot,  q_dpot)
            engine.Put(var_potm0, q_potm0)
            engine.Put(var_apars, q_apars)
            engine.Put(var_rz, self.mesh.rz)
            engine.Put(var_r, self.mesh.r)
            engine.Put(var_z, self.mesh.z)
            engine.Put(var_c, self.mesh.nd_connect_list)
            engine.EndStep()
            engine.Close()

        # Turbulence intensity
        if len(self.diag3D.Time) == 0:
            psimesh = self.mesh.psi / self.dataunits.psi_x
            self.diag3D.Mask = np.nonzero( (psimesh > self.diag3D.psirange[0]) & (psimesh < self.diag3D.psirange[1]) & (self.mesh.rz[:, 1] > self.dataunits.eq_x_z) )
            self.diag3D.Mask = self.diag3D.Mask[0]
            self.diag3D.enp = np.empty( shape=(0, 0) )
            self.diag3D.enn = np.empty( shape=(self.data3D.dpot.shape[0], 0))
        #var1 = self.data3D.dpot - np.mean(self.data3D.dpot, axis=0)
        var1 = self.data3D.dpot
        var1 = var1 / self.f0mesh.f0_T_ev[0, :]
        varsqr = var1 * var1
        # Time
        self.diag3D.Time += [self.data3D.time * 1E3]  # ms
        
        '''
        # Not using yet
        #s = np.mean(varsqr * self.mesh.node_vol) / np.mean(self.mesh.node_vol)
        #en = np.append(en, s)
        # partial sum
        sp = np.mean(varsqr[:, self.diag3D.Mask] * self.mesh.node_vol[self.diag3D.Mask]) / np.mean(self.mesh.node_vol[self.diag3D.Mask])
        self.diag3D.enp = np.append(self.diag3D.enp, sp)
        self.diag3D.ax.semilogy(self.diag3D.Time, np.sqrt(self.diag3D.enp))
        self.diag3D.ax.set_xlabel('Time (ms)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('$\sqrt{<(\phi/T_0)^2>}$', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "enp.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.ax.cla()
        '''

        # spectral
        vft = np.abs(np.fft.fft(var1, axis=0))**2
        sn = np.mean(vft[:, self.diag3D.Mask] * self.mesh.node_vol[self.diag3D.Mask], axis=1) / np.mean(self.mesh.node_vol[self.diag3D.Mask])
        sn = sn[:, np.newaxis]
        self.diag3D.enn = np.append(self.diag3D.enn, sn, axis=1)
        for i in range(0, self.diag3D.nmodes):
            self.diag3D.ax.semilogy(self.diag3D.Time, np.sqrt(self.diag3D.enn[i, :]) / self.data3D.dpot.shape[0], label='n={0}'.format(i * self.dataunits.sml_wedge_n))
            self.diag3D.ax.set_xlabel('Time (ms)', fontsize=self.diag3D.fontsize)
            self.diag3D.ax.set_ylabel('$\sqrt{<|\phi_n/T_0|^2>}$', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.legend(**self.diag3D.legend)
        imagename = os.path.join(outdir, "enn.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.ax.cla()

        self.diag3D.DashboardSave("mesh", self.data3D.timestep)
    """


    def Plot3D(self):

        # Directory setup (for dashboard)
        outdir = os.path.join("mesh-images", "{0}".format(self.data3D.timestep), "{0}-mesh".format(self.diag3D.codename))
        self.diag3D.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if ('write-adios' in self.diag3D.options) and self.diag3D.options['write-adios']:
            self.diag3D.group_name = "diag-mesh"
            ioname = "{1}.{0:.6f}".format(self.data3D.time*1e3, self.diag3D.group_name)
            self.diag3D.ADIOSFilename = os.path.join(outdir, "{0}.bp".format(ioname))
            self.diag3D.io = self.adios.DeclareIO(ioname)
            self.diag3D.engine = self.diag3D.io.Open(self.diag3D.ADIOSFilename, adios2.Mode.Write)
            self.diag3D.engine.BeginStep()
            self.diag3D.JsonList = []

        # Normalized dpot
        q_dpot = (self.data3D.dpot[self.diag3D.plane, :] - np.mean(self.data3D.dpot, axis=0)) / self.f0mesh.f0_T_ev[0, :]
        '''
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_dpot, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$\Phi/T_e$ w/o n=0 at %2.4f ms' % (self.data3D.time * 1e3) 
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "dpot.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])
        '''
        self.diag3D.PlotTriColorADIOS('r (m)', 'z (m)', '$\Phi/T_e$ w/o n=0 at {0:.4f} ms'.format(self.data3D.time * 1e3), self.diag3D.triobj, q_dpot, self.mesh.rz, "mesh_rz", self.mesh.nd_connect_list, "mesh_conn", "q_dpot", time=self.data3D.time)

        # pot n=0 mode
        q_potm0 = self.data3D.potm0
        '''
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_potm0, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$\Phi$ n=0 at %2.4f ms' % (self.data3D.time * 1e3)
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "potm0.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])
        '''
        self.diag3D.PlotTriColorADIOS('r (m)', 'z (m)', '$\Phi$ n=0 at {0:.4f} ms'.format(self.data3D.time * 1e3), self.diag3D.triobj, q_potm0, self.mesh.rz, "mesh_rz", self.mesh.nd_connect_list, "mesh_conn", "q_potm0", time=self.data3D.time)

        # Normalized Apar
        imass = self.dataunits.ptl_ion_mass_au * 1.67E-27
        vth_f0 = np.sqrt(1.6E-19 * self.f0mesh.f0_T_ev[0, :] / imass)
        q_apars = self.data3D.apars[self.diag3D.plane, :] * vth_f0 / self.f0mesh.f0_T_ev[0, :]
        '''
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_apars, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$A_{||} C_s/T_e$ at %2.4f ms' % (self.data3D.time * 1e3) 
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "apars.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])
        '''
        self.diag3D.PlotTriColorADIOS('r (m)', 'z (m)', '$A_{{||}} C_s/T_e$ at {0:.4f} ms'.format(self.data3D.time * 1e3), self.diag3D.triobj, q_apars, self.mesh.rz, "mesh_rz", self.mesh.nd_connect_list, "mesh_conn", "q_apars", time=self.data3D.time)

        '''
        if ('write-adios' in self.diag3D.options) and self.diag3D.options['write-adios']:
            var_dpot  = self.diag3D.io.DefineVariable("q_dpot",  q_dpot,  q_dpot.shape,  [0], q_dpot.shape)
            var_potm0 = self.diag3D.io.DefineVariable("q_potm0", q_potm0, q_potm0.shape, [0], q_potm0.shape)
            var_apars = self.diag3D.io.DefineVariable("q_apars", q_apars, q_apars.shape, [0], q_apars.shape)
            var_rz = self.diag3D.io.DefineVariable("mesh_rz", self.mesh.rz, self.mesh.rz.shape, [0,0], self.mesh.rz.shape)
            var_r = self.diag3D.io.DefineVariable("mesh_r", self.mesh.r, self.mesh.r.shape, [0], self.mesh.r.shape)
            var_z = self.diag3D.io.DefineVariable("mesh_z", self.mesh.z, self.mesh.z.shape, [0], self.mesh.z.shape)
            var_c = self.diag3D.io.DefineVariable("mesh_connectivity", self.mesh.nd_connect_list, self.mesh.nd_connect_list.shape, [0, 0], self.mesh.nd_connect_list.shape)
            self.diag3D.engine.BeginStep()
            self.diag3D.engine.Put(var_dpot,  q_dpot)
            self.diag3D.engine.Put(var_potm0, q_potm0)
            self.diag3D.engine.Put(var_apars, q_apars)
            self.diag3D.engine.Put(var_rz, self.mesh.rz)
            self.diag3D.engine.Put(var_r, self.mesh.r)
            self.diag3D.engine.Put(var_z, self.mesh.z)
            self.diag3D.engine.Put(var_c, self.mesh.nd_connect_list)
        '''

        # Turbulence intensity
        if len(self.diag3D.Time) == 0:
            psimesh = self.mesh.psi / self.dataunits.psi_x
            self.diag3D.Mask = np.nonzero( (psimesh > self.diag3D.psirange[0]) & (psimesh < self.diag3D.psirange[1]) & (self.mesh.rz[:, 1] > self.dataunits.eq_x_z) )
            self.diag3D.Mask = self.diag3D.Mask[0]
            self.diag3D.enp = np.empty( shape=(0, 0) )
            self.diag3D.enn = np.empty( shape=(self.data3D.dpot.shape[0], 0))
        #var1 = self.data3D.dpot - np.mean(self.data3D.dpot, axis=0)
        var1 = self.data3D.dpot
        var1 = var1 / self.f0mesh.f0_T_ev[0, :]
        varsqr = var1 * var1
        # Time
        #self.diag3D.Time += [self.data3D.time * 1E3]  # ms
        self.diag3D.Time = np.append(self.diag3D.Time, self.data3D.time * 1E3)  # ms
        
        '''
        # Not using yet
        #s = np.mean(varsqr * self.mesh.node_vol) / np.mean(self.mesh.node_vol)
        #en = np.append(en, s)
        # partial sum
        sp = np.mean(varsqr[:, self.diag3D.Mask] * self.mesh.node_vol[self.diag3D.Mask]) / np.mean(self.mesh.node_vol[self.diag3D.Mask])
        self.diag3D.enp = np.append(self.diag3D.enp, sp)
        self.diag3D.ax.semilogy(self.diag3D.Time, np.sqrt(self.diag3D.enp))
        self.diag3D.ax.set_xlabel('Time (ms)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('$\sqrt{<(\phi/T_0)^2>}$', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "enp.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.ax.cla()
        '''

        # spectral
        vft = np.abs(np.fft.fft(var1, axis=0))**2
        sn = np.mean(vft[:, self.diag3D.Mask] * self.mesh.node_vol[self.diag3D.Mask], axis=1) / np.mean(self.mesh.node_vol[self.diag3D.Mask])
        sn = sn[:, np.newaxis]
        self.diag3D.enn = np.append(self.diag3D.enn, sn, axis=1)
        '''
        for i in range(0, self.diag3D.nmodes):
            self.diag3D.ax.semilogy(self.diag3D.Time, np.sqrt(self.diag3D.enn[i, :]) / self.data3D.dpot.shape[0], label='n={0}'.format(i * self.dataunits.sml_wedge_n))
            self.diag3D.ax.set_xlabel('Time (ms)', fontsize=self.diag3D.fontsize)
            self.diag3D.ax.set_ylabel('$\sqrt{<|\phi_n/T_0|^2>}$', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.legend(**self.diag3D.legend)
        imagename = os.path.join(outdir, "enn.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.ax.cla()
        '''
        enns = []
        labels = []
        names = []
        for i in range(0, self.diag3D.nmodes):
            enns += [np.sqrt(self.diag3D.enn[i, :]) / self.data3D.dpot.shape[0]]
            labels += ['n={0}'.format(i * self.dataunits.sml_wedge_n)]
            names += ["enn={0}".format(i)]
        self.diag3D.PlotLinesADIOS('Time (ms)', '$\sqrt{<|\phi_n/T_0|^2>}$', self.diag3D.Time, enns, labels, "time", names, logy=True, time=self.data3D.time)
        
        if ('write-adios' in self.diag3D.options) and self.diag3D.options['write-adios']:
            self.diag3D.engine.EndStep()
            self.diag3D.engine.Close()
            JsonFilename = os.path.join(outdir, "plots.json")
            with open(JsonFilename, "w") as outfile:
                outfile.write(json.dumps(self.diag3D.JsonList, indent=4))

        self.diag3D.DashboardSave("mesh", self.data3D.timestep)


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

        time = self.data3D.time * 1E3  # ms

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

        outdir = os.path.join(self.outdir, PlaneObj.outdir, "{0:05d}".format(self.data3D.timestep))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        imagename = os.path.join(outdir, "{1}.{0}".format(PlaneObj.ext, PlaneObj.outdir))
        PlaneObj.fig.savefig(imagename, bbox_inches="tight")
        PlaneObj.fig.clear()
        if PlaneObj.movie and (kittie is not None):
            PlaneObj.PlotMovie.AddFrame(os.path.abspath(imagename))
        if PlaneObj.dashboard and (kittie is not None):
            PlaneObj.DashboardSave(PlaneObj.outdir, self.data3D.timestep, os.path.dirname(imagename))


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

        self.TurbData.Time += [self.data3D.time * 1E3]  # ms
      
        outdir = os.path.join(self.outdir, self.TurbData.outdir, "{0:05d}".format(self.data3D.timestep))
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
            self.TurbData.DashboardSave(self.TurbData.outdir, self.data3D.timestep, os.path.dirname(imagename))


    class BaseData(object):

        def __init__(self, filename, ioname, adios, perstep=None, subsample_factor=1, skip=0):
            self.filename = filename
            self.ioname = ioname
            self.on = False
            self.variables = []
            self.adios = adios
            self.perstep = perstep
            self.subsample_factor = subsample_factor
            self.period = None
            self.skip = skip
            self.StepCount = 0

            self.PerStepIndexes = None
            self.PerStepIndex = 0
            self.LastIndexed = False
            self.timeout = 0.0
            self.opened = False


        def AddVariables(self, variables):
            self.on = True
            for variable in variables:
                if variable not in self.variables:
                    self.variables += [variable]

        """
        def FindPeriod(self, pattern=None):
            if pattern is not None:
                self.period = GetText(os.path.join(os.path.dirname(self.filename), "input"), pattern)
                if self.period is not None:
                    self.period = int(self.period)
        """


        def GetFileMatches(self):
            matches = []
            results = os.listdir(os.path.dirname(self.filename))
            prefix = os.path.basename(self.filename)[:-2]
            for result in results:
                if result.startswith(prefix) and result.endswith(".bp"):
                    matches += [result]
            return matches


        def AddPerStepIndexes(self, indexes, leq, geq=None):
            if geq is None:
                geq = int(self.PerStepIndexes[-1]) + 1
            indexes_int = np.int32(indexes)
            cut = (indexes_int >= geq) & (indexes_int <= leq)
            indexes_use = indexes[cut]
            self.PerStepIndexes = np.append(self.PerStepIndexes, indexes_use)
            if np.sum(indexes_int >= leq) > 0:
                self.LastIndexed = True


        def PerStepCheck(self, StartTime, EndTime):
            if self.perstep is None:
                matches = self.GetFileMatches()
                if len(matches) > 1:
                    self.perstep = True
                elif len(matches) == 1:
                    if matches[0] == os.path.basename(self.filename):
                        self.perstep = False
                    else:
                        self.perstep = True
            if self.perstep:
                matches = self.GetFileMatches()
                indexes = SortedIndexes(matches)
                """
                geq = None
                if self.PerStepIndexes is None:
                    self.PerStepIndexes = np.empty(0, dtype=indexes.dtype)
                    geq = StartTime
                self.AddPerStepIndexes(indexes, EndTime, geq=geq)
                if (len(self.PerStepIndexes) > 0) and (self.PerStepIndex < len(self.PerStepIndexes)):
                    self.newfilename = "{0}.{1}.bp".format(self.filename[:-3], self.PerStepIndexes[self.PerStepIndex])
                """
                if self.PerStepIndexes is None:
                    self.PerStepIndexes = indexes
                else:
                    cut = (np.int32(indexes) > int(self.PerStepIndexes[-1]))
                    self.PerStepIndexes = np.append(self.PerStepIndexes, indexes[cut])
                if (len(self.PerStepIndexes) > 0) and (self.PerStepIndex < len(self.PerStepIndexes)):
                    self.newfilename = "{0}.{1}.bp".format(self.filename[:-3], self.PerStepIndexes[self.PerStepIndex])
            else:
                self.newfilename = self.filename


        def PerStepClose(self):
            if self.perstep:
                self.engine.Close()
                self.opened = False
                self.adios.RemoveIO(self.ioname)
                self.PerStepIndex += 1


        def NewDataCheck(self, dt=None, StartTime=None, EndTime=None):
            if self.LastIndexed:
                self.Stop()
                return False
            
            self.PerStepCheck(StartTime, EndTime)
            if (self.perstep and (self.PerStepIndex < len(self.PerStepIndexes))) or (not self.perstep and not self.opened):
                self.io = self.adios.DeclareIO(self.ioname)
                self.engine = self.io.Open(self.newfilename, adios2.Mode.Read)
                self.opened = True
                """
                if self.perstep and (dt is not None):
                    self.timestep = int(self.PerStepIndexes[self.PerStepIndex])
                    self.time = self.timestep * dt
                """
            '''
            elif self.perstep and (self.PerStepIndex >= len(self.PerStepIndexes)):
                if self.LastIndexed:
                    self.Stop()
                return False
            '''

            status = self.engine.BeginStep(adios2.StepMode.Read, self.timeout)
            
            if (status == adios2.StepStatus.OK):
                #self.StepCount += 1
                #if (dt is not None) and (self.StepCount <= self.skip):
                if (dt is not None) and (self.StepCount < self.skip):
                    self.engine.EndStep()
                    NewData = False
                    self.PerStepClose()
                #elif not self.perstep and (dt is not None):
                elif dt is not None:
                    print("dt: {0}, {1}".format(dt, self.newfilename))
                    var = self.io.InquireVariable("time")
                    try:
                        self.time = np.empty(1, dtype=GetType(var))
                    except:
                        self.engine.EndStep()
                        NewData = False
                        return NewData
                    self.engine.Get(var, self.time, adios2.Mode.Sync)
                    self.time = self.time[0]
                    self.timestep = round(self.time / dt) + 1
                    if self.timestep < StartTime:
                        self.engine.EndStep()
                        NewData = False
                        self.PerStepClose()
                    elif (self.timestep >= StartTime) and (self.timestep < EndTime):
                        NewData = True
                    elif self.timestep == EndTime:
                        NewData = True
                        self.LastIndexed = True
                    elif self.timestep > EndTime:
                        self.engine.EndStep()
                        NewData = False
                        self.LastIndexed = True
                        self.PerStepClose()
                        self.Stop()
                else:
                    NewData = True
                self.StepCount += 1
            elif (status == adios2.StepStatus.NotReady):
                NewData = False
                '''
                if not self.perstep and (dt is not None) and (self.timestep >= EndTime):
                    self.Stop()
                '''
            elif (status == adios2.StepStatus.EndOfStream):
                print(self.newfilename, "EndOfStream"); sys.stdout.flush()
                NewData = False
                self.Stop()
            elif (status == adios2.StepStatus.OtherError):
                NewData = False
                print("1D data file {0} encountered an error in BeginStep -- closing the stream and aborting its usage".format(self.newfilename), file=sys.stderr)
                self.Stop()
               
            return NewData


        def SingleRead(self):
            while not self.NewDataCheck(dt=None):
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

            self.engine.EndStep()
  
            self.PerStepClose()
            '''
            if self.perstep:
                self.engine.Close()
                self.opened = False
                self.adios.RemoveIO(self.ioname)
                self.PerStepIndex += 1
            '''


        def Stop(self):
            if self.opened:
                self.engine.Close()
            self.opened = False
            self.on = False


