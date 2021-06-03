
[SUAVE: An Aerospace Vehicle Environment for Designing Future Aircraft](http://suave.stanford.edu)
=======


SUAVE is a multi-fidelity conceptual design environment.
Its purpose is to credibly produce conceptual-level design conclusions
for future aircraft incorporating advanced technologies.

[![Build status](https://ci.appveyor.com/api/projects/status/h33v9tottm2t5b9a?svg=true)](https://ci.appveyor.com/project/planes/suave)
[![Coverage Status](https://coveralls.io/repos/github/suavecode/SUAVE/badge.svg?branch=develop)](https://coveralls.io/github/suavecode/SUAVE?branch=develop)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4784705.svg)](https://doi.org/10.5281/zenodo.4784705)

License: LGPL-2.1

Guides and Forum available at [suave.stanford.edu](http://suave.stanford.edu).


Contributing Developers
-----------------------
* Andrew Wendorff
* Anil Variyar
* Carlos Ilario
* Emilio Botero
* Francisco Capristan
* Jordan Smart
* Juan Alonso
* Luke Kulik
* Matthew Clarke
* Michael Colonno
* Michael Kruger
* Michael Vegh
* Pedro Goncalves
* Racheal Erhard
* Rick Fenrich
* Tarik Orra
* Theo St. Francis
* Tim MacDonald
* Tim Momose
* Tom Economon
* Trent Lukaczyk
* Walter Maier

Contributing Institutions
-------------------------
* Stanford University Aerospace Design Lab ([adl.stanford.edu](http://adl.stanford.edu))
* Embraer ([www.embraer.com](http://www.embraer.com))
* NASA ([www.nasa.gov](http://www.nasa.gov))

Simple Setup
------------

```
git clone https://github.com/suavecode/SUAVE.git
cd SUAVE/trunk
python setup.py install
```

More information available at [download](http://suave.stanford.edu/download.html).


Requirements
------------

numpy, scipy, matplotlib, pip, scikit-learn


Developer Install
-----------------

See [develop](http://suave.stanford.edu/download/develop_install.html).

Citing SUAVE
-----------------

This respository may be cited via BibTex as:

```
@software{SUAVEGit,
  author = {
    Wendorff, A. and
    Variyar, A. and
    Ilario, C. and
    Botero, E. and
    Capristan, F. and
    Smart, J. and 
    Alonso, J. and
    Kulik, L. and
    Clarke, M. and
    Colonno, M. and 
    Kruger, M. and
    Vegh, J. M. and 
    Goncalves, P. and
    Erhard, R. and
    Fenrich, R. and
    Orra, T. and 
    St. Francis, T. and
    MacDonald, T. and
    Momose, T. and
    Economon, T. and
    Lukaczyk, T. and
    Maier, W.
},
  title = {SUAVE: An Aerospace Vehicle Environment for Designing Future Aircraft},
  url = {https://github.com/suavecode/SUAVE},
  version = {2.1},
  year = {2020},
}
```
The most recent publication covering the general capabilities of SUAVE was presented at the 18th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference and may be cited via BibTex as:

```
@inbook{SUAVE2017,
author = {Timothy MacDonald and Matthew Clarke and Emilio M. Botero and Julius M. Vegh and Juan J. Alonso},
title = {SUAVE: An Open-Source Environment Enabling Multi-Fidelity Vehicle Optimization},
booktitle = {18th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference},
chapter = {},
pages = {},
doi = {10.2514/6.2017-4437},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2017-4437},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2017-4437}
}
```
