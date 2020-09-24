# ODNTD
This is the source code of the paper named **"Implanting Domain Knowledge into Feature Selection for Effective Outlier Detection in Network Traffic Data"** .


## Dataset 
- KDD(n)
- IDS12-Sat
- IDS12-Sun
- IDS17-Wa
- IDS17-Ds
Please download the all datasets from https://github.com/wilburwang520/ODNTD/.


## Dependencies
```
Python 3.6
Tensorflow == 1.12.0
pandas == 0.23.0
scikit-learn == 0.19.1
numpy == 1.14.3
```

## To run ONDTD
1. run main.py for sample usage.  
2. Data set format: the name of categorical attributes should be named as "A1", "A2", ..., and the numerical ones are "B1", "B2", ...  
3. The input path can be an individual data set or just a folder.  
