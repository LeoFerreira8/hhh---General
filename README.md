This is a program to calculate the parameter space for 2HDM models that are accepted by theoretical and experimental constraints. This was the base code used in https://arxiv.org/abs/2411.00094.

Instructions:

The main program to run is *scan_parameterspace.py*. 
In the file *scan_parameterspace_funcs.py* there are some options that can be changed, such as the intervals considered, if the analysis is restricted to the alignment or not, if l_5 is to be taken small or not (and what is the maximum value) and what is the type of 2HDM.
Moreover, in this file there is also the values of the constant used.

The auxiliary files are *scan_SPheno_funcs.py*, *scan_higgs_tools.py* and *bsg.py* for SPheno and Higgstools analysis and b->s gamma bounds respectively.

If you use this code for your purposes, please cite the original reference (https://arxiv.org/abs/2411.00094)
