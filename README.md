# Python Implementation of LEML

Python implementation of the LEML: 

Yu, Hsiang-Fu, et al. "[Large-scale Multi-label Learning with Missing Labels.](http://www.jmlr.org/proceedings/papers/v32/yu14.pdf)" ICML. 2014.

**Notes**: This version implements LEML with the **squared loss** and assuming **full** labels.

# How to Install

Packages:
- Python 2.7
- numpy 1.11.1+
- scipy 0.18.0+
- Cython
- OpenMP

To install the main package:

    python setup.py install

# How to Use

See examples in example directory.

# TODO

- Add ability to model missing label
- Add log loss function
