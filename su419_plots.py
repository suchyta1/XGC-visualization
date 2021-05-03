#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import adios2
import matplotlib

import yaml
import insitu_reader


def SetOptions(options):
    known = ['turbulence intensity']

    for name in known:
        if name not in options:
            options[name] = {'on': True}
        elif 'on' not in options[nam]:
            options[name]['on'] = True
            
    return options
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", help="Path to the data directory")
    parser.add_argument("-c", "color", help="Color to split with", type=int, default=10)
    parser.add_argument("-f", "optsfile", help="Plot options file", type=str, default=None)
    args = parser.parse_args()

    options = {}
    if os.path.exits(args.optsfile):
        with open(args.optsfile, 'r') as ystream:
            options = yaml.load(ystream, Loader=yaml.FullLoader)
    options = SetOptions(options)

    adiosargs = []
    xmlfile = os.path.join(args.datadir, "adios2cfg.xml")
    if os.path.exists(xmlfile):
        adiosargs += [xmlfile]
    adios = adios.ADIOS(*adiosargs)

    xgc1 = insitu_reader.xgc1(options)

    while xgc1.NotDone()
        xgc1.MakePlots()

