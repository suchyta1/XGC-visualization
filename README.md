# XGC-visualization

Python script for XGC data analysis

```
usage: xgc-visualization.py [-h] [-c COLOR] [-f OPTSFILE] datadir

positional arguments:
  datadir               Path to the XGC data directory

optional arguments:
  -h, --help            show this help message and exit
  -f OPTSFILE, --optsfile OPTSFILE
                        Plot options file (YAML)
```

Parameter defaults for `--optsfile`:

```yaml
subsample-factor-3D: 1
last-step: Null

turbulence intensity:
  use: true
  psirange: [0.17, 0.4]
  nmodes: 9
  fontsize: medium
  movie: false
  legend:
    ncol: 2
    loc: best
    fontsize: medium

dphi:
  use: true
  plane: 0
  cmap: jet
  levels: 50
  percentile: Null
  fontsize: medium
  movie: false

dA:
  use: true
  plane: 0
  cmap: jet
  levels: 50
  percentile: Null
  fontsize: medium
  movie: false
```
