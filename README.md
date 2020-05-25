## ts2sls_python


Ankit Bhutani

This is a Python program to calculate two-sample two-stage least squares (ts2sls) estimates based on Inoue and Solon (2010).
This is a follow up to Dr. Jeffrey Shrader's Stata code for the same available [here](https://github.com/jshrader/ts2sls).

As a benchmark, I have also included a Python program to calculate 2sls estimates which are equivalent to Stata's ivregress 2sls.


### Syntax for Python
```
from ivregress import ts2sls
ts2sls(S1, S2, y_var, regs, ev, inst)
```

where:
* S1 is the dataframe with sample 1 - this may not have the endogenous variable
* S2 is the dataframe with sample 2 - this may not have the dependent variable
* y_var is the name of dependent variable
* regs is the list of regressors or indepdent variables
* ev is endogenous variable provided as a list with 1 element
* inst is the list of exogenous instruments

For some examples, see unit_tests.py

### Installation
There are multiple possible ways. I have provided 2 simple ones here
1. For a quick and dirty solution, you can download the ivregress.py file to the folder where you have your Python script (example - unit_tests.py) to call the ts2sls function. This would enable Python to find the functions easily.
1. A more permanant solution requires you to find where your Python modules are stored. Run this code on your terminal `python -c "import sys; print(sys.path)" `. This would print a list of locations that Python searches for a module to be imported. Look for the one which ends in site-packges (example - /opt/anaconda3/lib/python3.7/site-packages). Download the file ivregress.py to this location.


### For R users
You can install reticulate library in R which allows you to call Python modules from R interface. For instructions on how to use reticulate, see [this](https://rstudio.github.io/reticulate/articles/calling_python.html). As invregress code calls pandas library, you may also need to install that. Use the following code to complete both these tasks:
```
install.packages("reticulate")
library(reticulate)
py_install("pandas")
```
Once the installation is complete, follow the example code - unit_tests.R to run these functions in R.


### References:
Inoue, Atsushi, and Gary Solon. "Two-sample instrumental variables estimators." The Review of Economics and Statistics 92, no. 3 (2010): 557-561.



