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
        self.mesh = self.BaseData(os.path.join(self.datadir, "xgc.mesh.bp"), "diagnosis.mesh", self.adios, perstep=False)
        self.volumes = self.BaseData(os.path.join(self.datadir, "xgc.volumes.bp"), "output.volumes", self.adios, perstep=False)
        self.data3D = self.BaseData(os.path.join(self.datadir, "xgc.3d.bp"), "field3D", self.adios, subsample_factor=options['subsample-factor-3D'], skip=options['diag3D']['skip'])
        self.data1D = self.BaseData(os.path.join(self.datadir, "xgc.oneddiag.bp"), "diagnosis.1d", self.adios, subsample_factor=options['subsample-factor-1D'], perstep=False, skip=options['diag1D']['skip'])

        self.dataunits.AddVariables(["sml_dt", "diag_1d_period", "sml_wedge_n", "eq_x_z", "eq_axis_r"])


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
            self.diag3D.Time = np.empty(0, dtype=np.float64)
            self.data3D.AddVariables(["dpot", "apars", "time"])
            self.mesh.AddVariables(["node_vol", "psi", "rz", "nd_connect_list"])



        # Start getting data

        self.dataunits.SingleRead()
        self.dataunits.psi_x = float(GetText(os.path.join(self.datadir, "units.m"), "^\s*psi_x\s*=\s*(.*)\s*;\s*$"))
        self.dataunits.ptl_ion_mass_au = float(GetText(os.path.join(self.datadir, "units.m"), "^\s*ptl_ion_mass_au\s*=\s*(.*)\s*;\s*$"))
        self.dataunits.diag_1d_period = int(self.dataunits.diag_1d_period)
        self.dataunits.sml_dt = float(self.dataunits.sml_dt)
        self.dataunits.sml_wedge_n = int(self.dataunits.sml_wedge_n)

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



    def NotDone(self):
        return self.data3D.on or self.data1D.on


    def Close(self):
        for diag in [self.diag3D, self.diag1D]:
            if diag.DashboardInit:
                diag.DashboardEngine.Close()


    def MakePlots(self):
        new3D = False
        new1D = False

        if self.data3D.on and self.data3D.NewDataCheck(dt=self.dataunits.sml_dt, StartTime=self.StartTime, EndTime=self.EndTime):
            self.data3D.GetData()
            self.data3D.time = self.data3D.time[0]
            new3D = True
            print("step 3D: {0}, {1}".format(self.data3D.timestep, self.data3D.newfilename)); sys.stdout.flush()

        if self.options['diag3D']['use'] and new3D:
            self.Plot3D()

        if self.data1D.on and self.data1D.NewDataCheck(dt=self.dataunits.sml_dt, StartTime=self.StartTime, EndTime=self.EndTime):
            self.data1D.GetData()
            self.data1D.time = self.data1D.time[0]
            new1D = True
            print("step 1D: {0} {1}".format(self.data1D.timestep, self.data1D.time)); sys.stdout.flush()

        if self.options['diag1D']['use'] and new1D:
            self.Plot1D()


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
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Elec. g.c. Density (m^-3)', self.data1D.psi, [self.data1D_0.e_gc_density_df_1d,self.data1D.e_gc_density_df_1d], ['Initial',"t={0:.2e} ms".format(self.data1D.time * 1e3)], "psi", ["e_gc_density_df_1d_0","e_gc_density_df_1d"], time=self.data1D.time)
 
        # Ion density
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Ion g.c. Density (m^-3)', self.data1D.psi, [self.data1D_0.i_gc_density_df_1d,self.data1D.i_gc_density_df_1d], ['Initial',"t={0:.2e} ms".format(self.data1D.time * 1e3)], "psi", ["i_gc_density_df_1d_0","i_gc_density_df_1d"], time=self.data1D.time)

        # Electron temperature
        Te = (self.data1D.e_perp_temperature_df_1d + self.data1D.e_parallel_mean_en_df_1d)/3*2
        Te_0 = (self.data1D_0.e_perp_temperature_df_1d + self.data1D_0.e_parallel_mean_en_df_1d)/3*2
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Elec. Temperature (eV)', self.data1D.psi, [Te_0,Te], ['Initial',"t={0:.2e} ms".format(self.data1D.time * 1e3)], "psi", ["Te_0","Te"], time=self.data1D.time)

        # Ion temperature
        Ti = (self.data1D.i_perp_temperature_df_1d + self.data1D.i_parallel_mean_en_df_1d)/3*2
        Ti_0 = (self.data1D_0.i_perp_temperature_df_1d + self.data1D_0.i_parallel_mean_en_df_1d)/3*2
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Ion Temperature (eV)', self.data1D.psi, [Ti_0,Ti], ['Initial',"t={0:.2e} ms".format(self.data1D.time * 1e3)], "psi", ["Ti_0","Ti"], time=self.data1D.time)

        # Ion parallel flow
        self.diag1D.PlotLinesADIOS('Normalized Pol. Flux', 'Ion. parallel flow (m/s)', self.data1D.psi, [self.data1D_0.i_parallel_flow_df_1d, self.data1D.i_parallel_flow_df_1d], ['Initial',"t={0:.2e} ms".format(self.data1D.time * 1e3)], "psi", ["i_parallel_flow_df_1d_0","i_parallel_flow_df_1d"], time=self.data1D.time)

        # Potential
        p0 = self.data1D_0.psi00/self.dataunits.psi_x
        p1 = self.data1D.psi00/self.dataunits.psi_x
        self.diag1D.PlotLinesADIOS('psi00/psix', 'Potential (V)', p1, [self.data1D_0.pot00_1d, self.data1D.pot00_1d], ['Initial',"t={0:.2e} ms".format(self.data1D.time * 1e3)], "psi00/psi_x", ["pot00_1d_0", "pot00_1d"], time=self.data1D.time)

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
        q_dpot = (self.data3D.dpot[self.diag3D.plane, :] - np.mean(self.data3D.dpot, axis=0))

        """
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_dpot, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$\Phi$ w/o n=0 at %.2e ms' % (self.data3D.time * 1e3) 
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "dpot.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])
        """
        
        self.diag3D.PlotTriColorADIOS('r (m)', 'z (m)', '$\Phi$ w/o n=0 at {0:.2e} ms'.format(self.data3D.time * 1e3), self.diag3D.triobj, q_dpot, self.mesh.rz, "mesh_rz", self.mesh.nd_connect_list, "mesh_conn", "q_dpot", time=self.data3D.time)

        # Normalized Apar
        q_apars = self.data3D.apars[self.diag3D.plane, :]

        """
        ColorAxis = self.diag3D.ax.tricontourf(self.diag3D.triobj, q_apars, cmap=self.diag3D.cmap, extend='both', levels=self.diag3D.levels)
        ColorBar = self.diag3D.fig.colorbar(ColorAxis, ax=self.diag3D.ax, pad=0)
        ColorBar.ax.tick_params(labelsize=self.diag3D.fontsize)
        ColorBar.ax.yaxis.offsetText.set_fontsize(self.diag3D.fontsize)
        title = '$A_{||}$ at %.2e ms' % (self.data3D.time * 1e3) 
        self.diag3D.ax.set_title("{0}".format(title), fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_aspect(1)
        self.diag3D.ax.set_xlabel('r (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.set_ylabel('z (m)', fontsize=self.diag3D.fontsize)
        self.diag3D.ax.tick_params(axis='both', which='major', labelsize=self.diag3D.fontsize)
        imagename = os.path.join(outdir, "apars.{0}".format(self.diag3D.ext))
        self.diag3D.fig.savefig(imagename, bbox_inches="tight")
        self.diag3D.fig.clear()
        self.diag3D.ax = self.diag3D.fig.add_subplot(self.diag3D.gs[0, 0])
        """
        
        self.diag3D.PlotTriColorADIOS('r (m)', 'z (m)', '$A_{{||}}$ at {0:.2e} ms'.format(self.data3D.time * 1e3), self.diag3D.triobj, q_apars, self.mesh.rz, "mesh_rz", self.mesh.nd_connect_list, "mesh_conn", "q_apars", time=self.data3D.time)


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


