# LEIS-Peak-Fitting-Dash-Visualization
This project is developed to custom fit functions over LEIS spectra obtained from experiments.

## Getting Started
Entire code has been developed using python 3.7 and [dash](https://dash.plotly.com/). 

You can install python either from [Anaconda](https://www.anaconda.com/products/individual) Recommended or only python from [python.org](https://www.python.org/)

For Visualization, dash can be installed via
```
pip install dash==1.13.3
```
in your command line as mentioned [here](https://dash.plotly.com/installation)

This code installs all the necessary dash related components. Other necessary modules are

1. numpy
2. pandas
3. ujson
4. xlrd
5. plotly
6. scipy
7. lmfit

All of the above libraries can be installed in a go by the following
```
pip install numpy pandas ujson xlrd plotly scipy lmfit
```

## Deployment

Copy this repo and save in your system. Run `dashapp.py` via 
```
python dashapp.py
```
on your terminal. And open a browser with URL http://127.0.0.1:8050/


## Usage

The app is intended to be used in the following manner
- Step-1: Choose the data file. If not already present, copy the file into the `/data` folder and restart the app.
- Step-2: Choose the corresponding fitting. Fit the data if the fitting is already not present. Restart the app once fitting process ends
- Step-3: Visualize the data via the indexed grid. Multiple selections can be made and `Clear all` button used to remove all graphs
- Step-4: [In progress] Export the rendered graph in step-3 to standalone HTML file or excel format

## TODO

Items remaining:
- CSS beautification of the app
- Fitting improvement
- Ternary relation to CSAF composition
- Export to HTML/Excel format