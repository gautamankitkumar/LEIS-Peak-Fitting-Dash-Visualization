# -*- coding: utf-8 -*-
import os
import sys 
import glob
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import ujson
import xlrd
import plotly.graph_objects as go
import time
from scipy.signal import find_peaks
from lmfit import Model


def read_data(filename):
    """ 
    This function reads the above specified file and returns two numpy array
    which contain the x-data and y-data 

    filename: string, Excel file to be read

    """
    
    # Iterate over all X-values. Y-values are stored in colummns of particular worksheet
    for x in range(0,13):

        wb = xlrd.open_workbook(filename)
        ws = wb.sheet_by_index(0)

        # This position of metadata doesn't change its relative position from sheet-to-sheet
        n_energy = int(ws.cell_value(1,3))
        n_iter = int(ws.cell_value(4,3))
        Rows_to_Skip = 15

        # Rename columns
        column_names = [str(x) for x in range(0,n_iter)]
        column_names.insert(0,'nan')
        column_names.insert(0,'KE')

        # Read data using pandas
        df_data = pd.read_excel(io = filename,
                           sheet_name=x,
                           skiprows = Rows_to_Skip,
                           names = column_names,
                           index_col='KE'
                          )
        # Drop the second column as it is always supposed to be false
        df_data.drop(columns=df_data.columns[0],inplace=True)
        
        # Get x_data as the index 
        x_array = np.array(df_data.index).reshape(len(df_data.index),1)
        
        # If we encounter first sheet
        if x==0:
            y = df_data.to_numpy()
            
        # Stack with the cummulative y built till now
        else:
            y = np.hstack((y, df_data.to_numpy()))
            
    # Ideally x_array should be (481, 1), and y should be (481, 169)
    return x_array, y

def gaussian(x, amp, cen, width):
    """1-d gaussian: gaussian(x, amp, cen, width)
    x: independent variable
    amp: amplitude/height of the curve
    cen: mean or peak position
    width: standard deviation/ spread of the curve
    """
    return (amp / (np.sqrt(2*np.pi) * width)) * np.exp(-(x-cen)**2 / (2*width**2))

def gauss2(x,a1,c1,w1,a2,c2,w2):
    """ For fitting two gaussian peaks """
    return gaussian(x,a1,c1,w1)+gaussian(x,a2,c2,w2)

def gauss3(x,a1,c1,w1,a2,c2,w2,a3,c3,w3):
    """ For fitting three gaussian peaks """
    return gaussian(x,a1,c1,w1)+gaussian(x,a2,c2,w2)+gaussian(x,a3,c3,w3)

# No need to re-run this again if data.json is already built
def fit_and_write_json(excel_file):
    """
    This function reads upon the excel file data and writes json file with fitted values
    """
    print(excel_file)
    # These variables are used subsequently in the code
    x_data,y_data = read_data(excel_file)
    
    # Create a dictionary to store peaks for now
    data = {}
    
    height = []
    for i in range(169):
        peaks,_ = find_peaks(y_data[:,i],height=5000,distance=50)
        data[i] = np.array(peaks,dtype=float)
    
    # Currently the dictionary should look like {'1': 1, '2': 2, '3':2 ...} and so on
    peak_data = data
    
    # Iterating over all 13 X and 13 Ys
    
    for i in range(169):
        
        # If scipy.signal.find_peaks finds only one peak
        if len(peak_data[i]) == 1:
            gmodel = Model(gaussian)
            peak = x_data[int(peak_data[i][0])][0]
            
            # Initialize appropriate singal from the peak data
            # center "c1" comes from the peak data itself
            c1 = peak
            
            if peak <= 850:
                a1 = 10000
                w1 = 20
            elif peak <= 900:
                a1 = 40000
                w1 = 15
            else:
                a1 = 80000
                w1 = 10
                
            # Fit using these initial estimates
            result = gmodel.fit(y_data[:,i], x=x_data[:,0],amp=a1,cen=c1,width=w1)
            y1 = gaussian(x_data,result.best_values['amp'],result.best_values['cen'],result.best_values['width'])
            new_dict = {'peak':1,'y':y_data[:,i].tolist(),'fit':result.best_fit.tolist(),
                        'y1':y1.tolist(),'mu1':result.best_values['cen']}
            
        elif len(peak_data[i]) == 2:
            # For two peaks
            peak1 = x_data[int(peak_data[i][0])][0]
            peak2 = x_data[int(peak_data[i][1])][0]
            
            c1 = peak1
            c2 = peak2
            if peak1<= 850:
                a1 = 10000
                w1 = 20
            elif peak1 <= 900:
                a1 = 40000
                w1 = 15
            else:
                a1 = 80000
                w1 = 10
                
            if peak2<= 850:
                a2 = 10000
                w2 = 20
            elif peak2 <= 900:
                a2 = 40000
                w2 = 15
            else:
                a2 = 80000
                w2 = 10
            # Fit two peaks
            gmodel = Model(gauss2)
            result = gmodel.fit(y_data[:,i], x=x_data[:,0], a1 = a1,c1=c1,w1=w1,a2=a2,c2=c2,w2=w2)
            y1 = gaussian(x_data[:,0],result.best_values['a1'],result.best_values['c1'],result.best_values['w1'])
            y2 = gaussian(x_data[:,0],result.best_values['a2'],result.best_values['c2'],result.best_values['w2'])
            new_dict = {'peak':2,'y':y_data[:,i].tolist(),'fit':result.best_fit.tolist(),
                        'y1':y1.tolist(),'y2':y2.tolist(),
                        'mu1':result.best_values['c1'],'mu2':result.best_values['c2']}
            
        else:
            peak1 = x_data[int(peak_data[i][0])][0]
            peak2 = x_data[int(peak_data[i][1])][0]
            peak3 = x_data[int(peak_data[i][2])][0]
            
            c1 = peak1
            c2 = peak2
            c3 = peak3
            
            if peak1<= 850:
                a1 = 25000
                w1 = 20
            elif peak1 <= 900:
                a1 = 25000
                w1 = 15
            else:
                a1 = 25000
                w1 = 10
                
            if peak2<= 850:
                a2 = 25000
                w2 = 20
            elif peak2 <= 900:
                a2 = 25000
                w2 = 15
            else:
                a2 = 25000
                w2 = 10
                
            if peak3<= 850:
                a3 = 25000
                w3 = 20
            elif peak3 <= 900:
                a3 = 25000
                w3 = 15
            else:
                a3 = 25000
                w3 = 10                
            
            # Fit three peaks
            gmodel = Model(gauss3)
            result = gmodel.fit(y_data[:,i], x=x_data[:,0], a1 = a1,c1=c1,w1=w1,a2=a2,c2=c2,w2=w2,a3=a3,c3=c3,w3=w3)
            y1 = gaussian(x_data[:,0],result.best_values['a1'],result.best_values['c1'],result.best_values['w1'])
            y2 = gaussian(x_data[:,0],result.best_values['a2'],result.best_values['c2'],result.best_values['w2'])
            y3 = gaussian(x_data[:,0],result.best_values['a3'],result.best_values['c3'],result.best_values['w3'])
            new_dict = {'peak':3,'y':y_data[:,i].tolist(),'fit':result.best_fit.tolist(),'y1':y1.tolist(),
                        'y2':y2.tolist(),'y3':y3.tolist(),
                        'mu1':result.best_values['c1'],'mu2':result.best_values['c2'],
                        'mu3':result.best_values['c3']}
        peak_data[i] = new_dict
    
    
    # At this point all the fitting is completed
    # Write the data into a json file
    new_file_name = 'fitted_data/fitted_'+excel_file[5:]+'.json'
    with open(new_file_name, 'w') as outfile:
        ujson.dump(peak_data, outfile)

import pickle
# # These variables are used subsequently in the code
# x_data,y_data = read_data('data/400 K LEIS 27.5.xlsx')
# print(x_data.shape,y_data.shape)

# with open('data/x_data', 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(x_data, f, pickle.HIGHEST_PROTOCOL)

# with open('data/y_data', 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(y_data, f, pickle.HIGHEST_PROTOCOL)


#Using pickle for data-source to speed up the loading process
with open('data/x_data', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    x_data = pickle.load(f)

with open('data/y_data', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    y_data = pickle.load(f)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the dictionary which contains checklist for grid points
grid_points = []
for i in range(169):
    grid_points.append({'label':str(i//13)+', '+str(i%13),'value':i})

# Define the dictionary which contains current directory's Excel files
dir_excel_files = []
for file in glob.glob("data/*.xlsx"):
    dir_excel_files.append(file)

excel_files_dict = []
for e in dir_excel_files:
    excel_files_dict.append({'label':str(e),'value':str(e)})

# Define the dictionary which contains current directory's Fitted json files
dir_fit_files = []
for file in glob.glob("fitted_data/*.json"):
    dir_fit_files.append(file)

fit_files_dict = []
fit_files_dict.append({'label':'None','value':0})
for f in dir_fit_files:
    fit_files_dict.append({'label':str(f),'value':str(f)})

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Reading Files', children=[
            html.Div([
            html.H3(children='List of all excel files (.xlsx) in data folder')
        ]),
        dcc.RadioItems(id='select_excel',options=excel_files_dict, value=str(dir_excel_files[0])) 
        ]),

        dcc.Tab(label='Generate/Choose Fit', children=[
            html.Div([
            html.H2('Currently fit available for following files'),
            html.H3(id='fit_state',children='Fit not available'),            
            html.Button(children='Fit now',id='fit-button',
            title='Fit the chosen excel file', n_clicks=0),
            ]),
            dcc.RadioItems(id='select_fit_file',options=fit_files_dict, value=str(dir_fit_files[0])),
            dcc.Loading(
                    id="loading_div",
                    children=[html.Div([html.Div(id="loading-output")])],
                    type="circle",
                    fullscreen=True,
                )
        ]),
        dcc.Tab(label='Visualize Data', children=[
            html.Div(children=[
                html.H3(id ='excel_file_name',children=""),
                dcc.Checklist(id='show-fit', options=[{'label':'Show Fitted lines','value':0}]),
                dcc.Checklist(id='index-grid', options=grid_points, value=[1]), 
                dcc.Graph(id='ternary-plot'),
                html.Button('Clear All',id='clear-button',
                title='Clears all the data graphs from the window', n_clicks=1,disabled=False),
                dcc.Graph(id='data-graph')
                ])]),
        dcc.Tab(label='Generate Fit', children=[
            html.Div([
                html.H1('Tab content 4')])])])])

n_fit = 0
@app.callback(
    Output("loading-output", "children"), 
    [Input("fit-button", "n_clicks"),
    Input('select_excel','value')])
def fit_new_data(fit_clicks,data_file):
    global n_fit
    if fit_clicks>n_fit:
        n_fit = fit_clicks
        fit_and_write_json(data_file)
        return 'Fitting Done'

@app.callback(
    Output('excel_file_name','children'),
    [Input('select_excel','value')]
)
def select_excel_to_plot(excel_name):
    return "Currently showing graph for {0}".format(excel_name)

@app.callback(
    [Output('fit_state','children'),
    Output('select_fit_file','value'),
    Output('fit-button','disabled')],
    [Input('select_excel','value')]
)
def fit_file(excel_file):
    fit_name = 'fitted_data\\fitted_' + excel_file[5:] + '.json'
    print(fit_name)
    if fit_name in dir_fit_files:
        return 'Fit available. Automatically choosing the fit file', fit_name, True
    else:
        return 'Fit not available. Press fit now to fit this new file. Then refresh page to see the fitted file', 0, False

@app.callback(
    Output('data-graph','figure'),
    [Input('index-grid','value'),
    Input('show-fit','value'),
    Input('select_excel','value')]
)
def update_graph(input_values,show_fit,excel_name):
    """ App callback function to read checked grid points and display the corresponding graph

    """
    fitted = new_file_name = 'fitted_data/fitted_'+excel_name[5:]+'.json'
    if os.path.isfile(fitted):
        with open(fitted) as json_file:
            peak_data = ujson.load(json_file)

        if not input_values:
            return {'data':[]}
        traces = []
        if show_fit:        
            for val in input_values:
                yfit = peak_data[str(val)]['fit']
                legendgroup_name = 'group'+str(val)
                traces.append(dict(
                x=np.array(x_data)[:,0].tolist(),
                y=np.array(y_data)[:,val].tolist(),
                text=str(val),
                mode='markers',
                opacity=0.5,
                legendgroup=legendgroup_name,
                name='Data PointsX:'+str(val//13) + ', Y:' + str(val%13),
                marker = dict(color=str(val)),
                    ))
                traces.append(dict(
                x=np.array(x_data)[:,0].tolist(),
                y=yfit,
                text=str(val),
                mode='line',
                opacity=0.7,
                legendgroup=legendgroup_name,
                name='Fitted, X:'+str(val//13) + ', Y:' + str(val%13),
                line = dict(color=str(val)),
                    ))
        else:        
            for val in input_values:
                traces.append(dict(
                x=np.array(x_data)[:,0].tolist(),
                y=np.array(y_data)[:,val].tolist(),
                text=str(val),
                mode='markers',
                opacity=0.7,
                name='Data Points X:'+str(val//13) + ', Y:' + str(val%13),
                    ))

        return {
            'data': traces,
            'layout': dict(
                xaxis={ 'title': 'KE'},
                yaxis={'title': 'Data points'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
            )
        }
    else:
        return {'data':[]}

current_n = 0
@app.callback(
    Output('index-grid','value'),
    [Input('clear-button','n_clicks')]
    )
def clear_graph(n_click):
    global current_n
    if n_click>current_n:
        current_n = n_click
        return []

@app.callback(
    Output('ternary-plot','figure'),
    [Input('index-grid','value')]
)
def get_ternary_plot(input_values):
    if not input_values:
        return {'data':[]}
    traces = []
    metal_1 = []
    metal_2 = []
    metal_3 = []
    for val in input_values:
        metal_1.append(val//13)
        metal_2.append(val%13)
    fig =  go.Figure(go.Scatterternary(text='Metal Composition',a=metal_1,b=metal_2,
    mode='markers',marker={'symbol': 100,'size': 10},))
    fig.update_layout({
    'title': 'Index Grid Points on CSAF',
    'ternary': {
        'sum': 39,
    'aaxis':{'title':'Component - A'},
    'baxis':{'title':'Component - B'},
    'caxis':{'title':'Component - C'},
    }})

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)