#!/usr/bin/env python

import re
import os
import sys
import argparse
import numpy as np
import adios2

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from scipy import stats
from scipy.signal import savgol_filter
from sklearn.neighbors import KernelDensity
import json


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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="Directory with heaload.bp")
    parser.add_argument("-f", "--factor", help="Factor that divides phi (number of processes heatload ran)", type=int, default=1)
    parser.add_argument("-p", "--planes", help="Plane selection", type=str, default=":")
    parser.add_argument("-c", "--cmap",   help="Color map", type=str, default="jet")
    args = parser.parse_args()

    args.planes = args.planes.split(",")
    for i, plane in enumerate(args.planes):
        args.planes[i] = plane.strip()

    FS = 24
    gs = gridspec.GridSpec(1, 2, height_ratios=[1], width_ratios=[1,1])
    data = {}
    heatfile = os.path.join(args.datadir, "xgc.heatload.bp")
   
    xml = "adios2cfg.xml"
    if os.path.exists(xml):
        adios = adios2.ADIOS(xml)
    else:
        adios = adios2.ADIOS()

    io = adios.DeclareIO("heatload")
    engine = io.Open(heatfile, adios2.Mode.Read)
    istep = -1
    
    plotname = "heatload"
    ioname = plotname + ".done"
    DashboardIO = adios.DeclareIO(ioname)
    VarNumber = DashboardIO.DefineVariable("Step", np.empty(0, dtype=np.int32), [], [], [])
    DashboardEngine = DashboardIO.Open(ioname + ".bp", adios2.Mode.Write)
    
    while True:
        status = engine.BeginStep(adios2.StepMode.Read, 0.0)

        if (status == adios2.StepStatus.NotReady):
            continue
        elif (status == adios2.StepStatus.EndOfStream):
            break
        elif (status != adios2.StepStatus.OK):
            print("Unknown error in file reading: {0}".format(heatfile), file=sys.stderr)
            sys.exit(1)
       
        istep += 1
        AvailableVariables = io.AvailableVariables()
        for name in AvailableVariables:
            var = io.InquireVariable(name)
            if name not in data:
                data[name] = np.zeros(var.Shape(), dtype=var.Type().rstrip("_t"))
            var.SetSelection([np.zeros(data[name].ndim, dtype=np.int64), data[name].shape])
            engine.Get(var, data[name])
        engine.EndStep()

        """
        if istep < 600:
            continue
        """
        print(istep); sys.stdout.flush()

        sml_dt = float(GetText(os.path.join(args.datadir, "units.m"), "^\s*sml_dt\s*=\s*(.*)\s*;\s*$"))
       
        appname = os.path.basename(os.getcwd())
        outdir = os.path.join("{0}-images".format(plotname), "{0}".format(istep+1), "{1}-{0}".format(plotname, appname))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fig = plt.figure(1, figsize=(10,6))
        fig.subplots_adjust(wspace=0, hspace=0, top=1.0, left=0, bottom=0, right=1.0)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])

        PlotIO = adios.DeclareIO("HeatLoadPlot")
        PlotEngineFilename = os.path.join(outdir, "HeatLoad-{0}.bp".format(istep+1))
        PlotEngine = PlotIO.Open(PlotEngineFilename, adios2.Mode.Write)
        var = PlotIO.DefineVariable("psi", data["psi"][i, :], data["psi"][i, :].shape, np.zeros(1, dtype=np.int64), data["psi"][i, :].shape)
        PlotEngine.Put(var, data["psi"][i, :])
        JsonList = []

        for name in data:
            if name == "psi":
                continue
            index = np.arange(data[name].shape[1], dtype=np.int64)
            planes = np.empty(0, dtype=np.int64)
            for sel in args.planes:
                exec("add = index[{0}]".format(sel))
                planes = np.append(planes, add)
            planes = np.unique(np.sort(planes))
            
            for i in range(2):
                PlotYs = []
                lim = [100, 0]
                attrname = "{0} side={1}".format(name, i)
                for p, plane in enumerate(planes):
                    if np.amax(data[name][i, plane, :]) > 0:

                        ax0.plot(data['psi'][i, :], data[name][i, plane, :], c=cm.get_cmap(args.cmap)(p/len(planes)), label="A{0}".format(plane))

                        cut = (data[name][i, plane, :] > 0)
                        P = data['psi'][i, :][cut]

                        """
                        norm =  np.sum(data[name][i, plane, :]) * (data['psi'][i, 1] - data['psi'][i, 0])
                        d = data[name][i, plane, :][cut]
                        kde = KernelDensity(kernel="gaussian", bandwidth=3*(data['psi'][i, 1] - data['psi'][i, 0])).fit(P.reshape((P.shape[0], 1)), sample_weight=d)
                        log_dens = kde.score_samples(data['psi'][i, :].reshape((data['psi'][i, :].shape[0], 1)))
                        ax1.plot(data['psi'][i, :], np.exp(log_dens)*norm, c=cm.get_cmap(args.cmap)(p/len(planes)), label="A{0}".format(plane))
                        #kernel = stats.gaussian_kde(data['psi'][i, :], weights=data[name][i, plane, :], bw_method="scott")
                        #ax1.plot(data['psi'][i, :], kernel(data['psi'][i, :])*norm, c=cm.get_cmap(args.cmap)(p/len(planes)), label="A{0}".format(plane))
                        """
                        
                        ax1.plot(data['psi'][i, :], np.fabs(savgol_filter(data[name][i, plane, :], 101, 3)), c=cm.get_cmap(args.cmap)(p/len(planes)), label="A{0}".format(plane))

                        if np.amin(P) < lim[0]:
                            lim[0] = np.amin(P)
                        if np.amax(P) > lim[1]:
                            lim[1] = np.amax(P)

                    else:
                        ax0.plot(data['psi'][i, :], data[name][i, plane, :], c=cm.get_cmap(args.cmap)(p/len(planes)), label="A{0}".format(plane))
                        ax1.plot(data['psi'][i, :], data[name][i, plane, :], c=cm.get_cmap(args.cmap)(p/len(planes)), label="A{0}".format(plane))
                    #kde = KernelDensity(kernel="gaussian").fit(data[name][i, plane, :])
                    #log_dens = kde.score_samples(data['psi'][i, :])
                    #ax.plot(data['psi'][i, :], np.exp(log_dens), c=cm.get_cmap(args.cmap)(p/len(planes)), label="A{0}".format(plane))
                  
                    PlotYs += ["{0} plane={1}".format(attrname, plane)]
                    var = PlotIO.DefineVariable(PlotYs[-1], data[name][i, plane, :], data[name][i, plane, :].shape, np.zeros(1, dtype=np.int64), data[name][i, plane, :].shape)
                    PlotEngine.Put(var, data[name][i, plane, :])
                    
                ax0.set_ylabel(name, fontsize=FS)
                ax0.legend(fontsize=FS)
                ax1.yaxis.set_ticklabels([])
                ax1.tick_params(axis='y', which='major', left=False)
                ax1.set_ylim(ax0.get_ylim())

                ax0.set_xlabel("$\Psi$", fontsize=FS)
                ax0.set_xlim(lim)
                ax0.set_title("Original", fontsize=FS)
                ax1.set_xlabel("$\Psi$", fontsize=FS)
                ax1.set_xlim(lim)
                ax1.set_title("Smoothed", fontsize=FS)

                #fig.savefig(os.path.join(outdir, "{0}-{1}-{2}.pdf".format(name, i, istep+1)), bbox_inches="tight")
                fig.savefig(os.path.join(outdir, "{0}-{1}-{2}.png".format(name, i, istep+1)), bbox_inches="tight")
                ax0.cla()
                ax1.cla()

                attribute = {
                        "type": "lines",
                        "x": "psi",
                        "y": PlotYs,
                        "yname": PlotYs,
                        "xlabel": "$\Psi$",
                        "ylabel": "Cumulative Heat Load"
                        }
                PlotIO.DefineAttribute(attrname, json.dumps(attribute))
                #JsonList += [{'file_name': PlotEngineFilename, 'attribute_name': attrname, 'group_name': "Heat Load", 'time': istep+1}]
                JsonList += [{'file_name': PlotEngineFilename, 'attribute_name': attrname, 'group_name': "Heat Load", 'time': (istep)*sml_dt}]

            #plt.close(fig)

        PlotEngine.Close()
        adios.RemoveIO("HeatLoadPlot")
        plt.close(fig)
            
        DashboardEngine.BeginStep()
        var = DashboardIO.InquireVariable("Step")
        DashboardEngine.Put(var, np.array([istep+1], dtype=np.int32))
        DashboardEngine.EndStep()

        JsonFilename = os.path.join(outdir, "plots.json")
        with open(JsonFilename, "w") as outfile:
            outfile.write(json.dumps(JsonList, indent=4))

    engine.Close()
    DashboardEngine.Close()

