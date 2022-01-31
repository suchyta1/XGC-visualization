#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import adios2
import matplotlib

import argparse
import yaml
import insitu_reader


def SetOptions(options):
    known = ['turbulence intensity', 'dphi', 'dA']
    for name in known:
        if name not in options:
            options[name] = {'use': True}
        elif 'use' not in options[name]:
            options[name]['use'] = True

    if 'subsample-factor-3D' not in options:
        options['subsample-factor-3D'] = 1
    if 'last-step' not in options:
        options['last-step'] = None
    if 'codename' not in options:
        options['codename'] = "effis"

    return options
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="Path to the XGC data directory")
    parser.add_argument("-f", "--optsfile", help="Plot options file (YAML)", type=str, default=None)
    args = parser.parse_args()

    options = {}
    if (args.optsfile is not None) and os.path.exists(args.optsfile):
        with open(args.optsfile, 'r') as ystream:
            options = yaml.load(ystream, Loader=yaml.FullLoader)
    options = SetOptions(options)

    xgc1 = insitu_reader.xgc1(args.datadir, options)

    while xgc1.NotDone():
        xgc1.MakePlots()
    print("Plotting complete")
    xgc1.Close()
    print("exiting")
