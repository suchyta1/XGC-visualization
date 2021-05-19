# XGC-visualization

Python script for XGC data analysis and Maplotlib visualization

## Dependencies

- Numpy
- Matplotlib
- ADIOS Python bindings
- Optional: EFFIS Python bindings -- needed for movie EFFIS hooks

## Usage

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
subsample-factor-3D: 1    # Only plot every nth output that used 3D data
last-step: Null           # Only needed if not staging, and 'input' file not found

turbulence intensity:     # Makes the enn and enp plots
  use: true               # Turn on/off
  psirange: [0.17, 0.4]   # psi limits for calculation
  nmodes: 9               # Number of nodes in plot
  fontsize: medium        # Plot font size, for everything but the legend
  movie: false            # Triggers EFFIS to make movie after run. (This script itself doesn't make the movie.)
  legend:
    ncol: 2               # Number of colums in legend
    loc: best             # Matplotlib location parameter for legend
    fontsize: medium      # Legend fontsize

dphi:                     # Plot dpot on one poloidal plane
  use: true               # Turn on/off
  plane: 0                # Which plane to plot
  cmap: jet               # Color map
  levels: 50              # Number of color levels
  percentile: Null        # Percentile cap (using absolute value) included in color range. (Null <=> 100, i.e. min/max)
  fontsize: medium        # Plot font size
  movie: false            # Triggers EFFIS to make movie after run. (This script itself doesn't make the movie.)

dA:                       # Plot Apars on one poloidal plane
  use: true               # Turn on/off
  plane: 0                # Which plane to plot
  cmap: jet               # Color map
  levels: 50              # Number of color levels
  percentile: Null        # Percentile cap (using absolute value) included in color range. (Null <=> 100, i.e. min/max)
  fontsize: medium        # Plot font size
  movie: false            # Triggers EFFIS to make movie after run. (This script itself doesn't make the movie.)
```

## Figure Examples

| enn | enp |
| --- | --- |
| <img src="https://user-images.githubusercontent.com/3419552/118796091-43975480-b869-11eb-8ac5-f1941dc2c8d6.png" width="500"> | <img src="https://user-images.githubusercontent.com/3419552/118796106-472adb80-b869-11eb-897f-ab13b1889573.png" width="500"> |

| dphi | dA |
| ---- | -- |
| <img src="https://user-images.githubusercontent.com/3419552/118795722-e3a0ae00-b868-11eb-942f-7fb5d84a3b57.png" width="500"> | <img src="https://user-images.githubusercontent.com/3419552/118795905-12b71f80-b869-11eb-9fb0-2ae107edde91.png" width="500"> |

