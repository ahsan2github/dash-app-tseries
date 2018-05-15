#!/usr/bin/env python3

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import requests
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler
import accumulate_data as accd
from pandas.tseries.offsets import BDay
from statsmodels.tsa.arima_model import ARIMA

api_key = "RU1K3VX6KM5GVHKJ"

def getData(sym="MSFT"):
    req_str = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + \
               sym + "&outputsize=full&" + "apikey=" + api_key
    data=requests.get(req_str)
    dataDic = json.loads(data.content)
    if data.status_code == 200: 
        meta_data = dataDic['Meta Data']
        time_series_dict = dataDic['Time Series (Daily)'];
        dates = list(time_series_dict.keys());
        df = pd.DataFrame()
        df['date'] = [pd.to_datetime(x) for x in dates]
        df['open'] = [np.float64(time_series_dict[x]['1. open']) for x in dates]
        df['high'] = [np.float64(time_series_dict[x]['2. high']) for x in dates]
        df['low'] = [np.float64(time_series_dict[x]['3. low']) for x in dates]
        df['close'] = [np.float64(time_series_dict[x]['4. close']) for x in dates]
        df['volume'] = [np.float64(time_series_dict[x]['5. volume']) for x in dates]
        df['symbol'] = meta_data['2. Symbol']
        df.index = df['date']
        df.dropna(axis=0, how='any')
        return data.status_code, df
    else:
        return data.status_code, pd.DataFrame()

def analysedData(req_str):
    df = pd.DataFrame()
    ind_dl = []
    time.sleep(2)
    #req_str = "https://www.alphavantage.co/query?function=" + ind_list[ii] + "&symbol=" + sym + \
    #     "&interval=daily&time_period=" + str(avg_period) + "&series_type=" + stype + "&apikey="+api_key
    data=requests.get(req_str)
    if data.status_code == 200:
        df_temp = pd.DataFrame()
        dataDic = json.loads(data.content)
        kys = list(dataDic.keys())
        meta_data = dataDic[kys[0]]
        time_series_dict = dataDic[kys[1]];
        dates = list(time_series_dict.keys());
        df_temp['date'] = [pd.to_datetime(x) for x in dates]
        inner_kys = list(time_series_dict[dates[0]].keys())
        df_temp['value'] = [np.float64(time_series_dict[x][inner_kys[0]]) for x in dates]
        df_temp['property'] = str(inner_kys[0])
        ind_dl.append(inner_kys[0])
        df = df.append(df_temp)
        df.dropna(axis=0, how='any')
    return ind_dl, df

app = dash.Dash()
server = app.server
app.scripts.config.serve_locally = True

co_options = ['msft', 'gogl', 'appl']
with open('symbol_list.json', mode='r') as f:
    sym_list = json.load(f)

with open('tech_options_dic.txt', 'r') as infile:
    tech_indicator_dic = json.load(infile)
tech_indicator_list = list(tech_indicator_dic.keys())

app.layout = html.Div([
dcc.Markdown('''
#### Plotly-DASH App Demo for Stock Price Prediction
This app downloads stock price data from [Alpha Vantage](https://www.alphavantage.co) server using the provided 
[REST APIs](https://www.alphavantage.co/documentation/). Unfortunately, Alpha Vantage Server often can be unresponsive.
In that case you may have to wait and try again. If the app can't download data from the server, you will only see
demo staight lines instead of actual graph of the data. Alpha Vantage also provides caluclated 
technical indicator data of 50 methods. This app will download the Technical indicator data series based on the
selection from dropdown menu. All available indicators are not provided in this demo app at this point. 
As for the prediction, Autoregressive Integrated Moving Average (ARIMA) and Recurrent Neural Networks can be 
used at this point. The Hybrid ARIMA-RNN method is not implemeneted yet. In the future this method may be 
implemented following G. Peter Zhang's [paper](https://www.sciencedirect.com/science/article/pii/S0925231201007020).
''', ),

    html.Div([

        html.Div([
            html.P('Ticker', style={'text-align':'center'}),
            dcc.Dropdown(
                id="co-select",
                options=[{
                    'label': str(kys) + ":  " + str(sym_list.get(kys)),
                    'value': kys
                } for kys in sym_list.keys()],
                value= 'MSFT' ),

            html.Br(),
            
#            html.P("Data Request Status"),
#            html.Div(id='co-select-echo', style={'color': '#20B2AA'}),
            dcc.Textarea(
                id='co-select-echo',
                readOnly=1,
                placeholder='',
                value='',
                style={'width': '100%', 'background-color': '#F5F5F5', 'opacity': 0.5}
            ),
            html.Br(),

            html.P('Data Series Type', style={'text-align':'left'}),
            dcc.RadioItems(
                id='tseries-select-radio',
                options=[
                    {'label': 'OPEN', 'value': 'open'},
                    {'label': 'CLOSE', 'value': 'close'},
                    {'label': 'LOW', 'value': 'low'},
                    {'label': 'HIGH', 'value': 'high'}
                ],
                value='close',
                labelStyle={'display': 'inline-block'},
                style={'align-content':'center'}
            ),

            html.Hr(),

            html.P('Technical Indicator', style={'text-align':'center'}),
            dcc.Dropdown(
                id="tech-ind-select",
                options=[{
                    'label': tech_indicator_dic[i][i],
                    'value': i
                } for i in tech_indicator_list],
                value=tech_indicator_list[0]),

            html.Br(),

            dcc.Textarea(
                id='tech-indicator-desc',
                readOnly=1,
                placeholder=' ',
                value=' ',
                style={'width': '100%', 'background-color': '#F5F5F5', 'opacity': 0.5}
            ),
            html.Br(),

            html.P("Forecast (days)", style={'text-align':'center'}),
            dcc.Slider(
                id='ndays-predict',
                min=1,
                max=3,
                value=2,
                marks={
                    1: {'label': '1', 'style': {'color': '#77b0b1'}},
                    2: {'label': '2'},
                    3: {'label': '3'},
                }
            ),

            html.Br(),

            dcc.RadioItems(
                id='model-select-radio',
                options=[
                    {'label': 'ARIMA', 'value': 'arima'},
                    {'label': 'ARIMA + ANN', 'value': 'arima_ann'},
                    {'label': 'RNN', 'value': 'rnn'}
                ],
                value='arima',
                labelStyle={'display': 'inline-block'}
            ),
            html.Br(),
            html.P(""),
            html.Div([            
                html.Button('Run Model', id='run-model-button', n_clicks = 0, type='submit'),
                ],
            style = {'width':'45%', 'display':'inline-block', 'background-color':'#F5F5F5'}
            ),   
            html.Div([
                html.Button('Plot Model Prediction', id='plot-model-button', n_clicks = 0, type='submit'),
                ],
            style = {'width':'50%', 'display':'inline-block', 'background-color':'#F5F5F5'}
            ),
            html.Br(),
            html.Br(),
            dcc.Textarea(
                id='run-model-echo',
                readOnly=1,
                placeholder='',
                value='Model runtime-outputs will be updated here. It may take a while.',
                style={'width': '100%', 'height':'200', 'background-color': '#F5F5F5', \
                  'opacity': 0.5, 'fontSize': 10}
            )
        ], className="three columns"),

        html.Div([
            dcc.Graph(id='raw-series', figure={'data': [{'y': [1, 2, 3]}]}),

            dcc.Slider(
                id='ndays-plot-raw',
                min=7,
                max=365,
                value=7,
                marks={
                    7: {'label': '1 W', 'style': {'color': '#17BECF'}},
                    28: {'label': '4 W ', 'style':{'color':'#17BECF'}},
                    60: {'label': '2 M', 'style': {'color': '#17BECF'}},
                    120: {'label': '4 M ', 'style': {'color': '#17BECF'}},
                    240: {'label': '8 M', 'style': {'color': '#17BECF'}},
                    365: {'label': '1 Y', 'style': {'color': '#17BECF'}},
                }
            ),
            html.Br(),
            html.Div([
                 html.Div([
                    dcc.Graph(id='raw-tech-tseries', figure={'data': [{'y': [1, 2, 3]}]}),
                    ], style={'height':200}, className='six columns'),
                 html.Div([
                    dcc.Graph(id='tech-modeled-tseries', figure={'data': [{'y': [1, 2, 3]}]}),
                    ], className='six columns')
            ], className="row")

        ], className="eight columns"),


    ], className="row"),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='dl-dataframe', style={'display': 'none'}),
    html.Div(id='tech-ind-list-data', style={'display': 'none'}),
    html.Div(id='model-run-output', style={'display': 'none'}),      
])

#external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
#                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
#                "//fonts.googleapis.com/css?family=Dosis:Medium",
#                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/62f0eb4f1fadbefea64b2404493079bf848974e8/dash-uber-ride-demo.css",
#                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]


#for css in external_css:
#    app.css.append_css({"external_url": css})




app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

# download data based on co-select dropdown menu and save data
# in hidden html Div 'dl-dataframe'
@app.callback(
    Output('dl-dataframe', 'children'),
    [Input('co-select', 'value')
])
def retrieve_data(co_select_val):
    statcode, data_ = getData(co_select_val)      
    return json.dumps({
    'statcode_raw':statcode, 
    'df':data_.to_json(orient='split', date_format='iso')
    })

# download technical indicator data based on download of raw data
@app.callback(
    Output('tech-ind-list-data', 'children'),
    [Input('dl-dataframe', 'children'),
    Input('co-select', 'value'),
    Input('tseries-select-radio', 'value'),
    Input('tech-ind-select', 'value')
])
def retrieve_techInd_data(dummy_data, co_select_val, tseries_type, tech_ind):
    kl = list(tech_indicator_dic[tech_ind].keys())
    opt_str=""
    for ii in range(1,len(kl)):
        opt_str += "&"+kl[ii]+ "=" + str(tech_indicator_dic[kl[0]][kl[ii]])

    req_str = "https://www.alphavantage.co/query?function="+tech_ind+"&symbol="+co_select_val+\
        "&series_type="+tseries_type + opt_str + "&apikey="+ api_key
    ind_list_ret, data_ = analysedData(req_str)    
    print(list(data_))
    return json.dumps({
    'ind_list': ind_list_ret, 
    'df_techInd':data_.to_json(orient='split', date_format='iso')
    })
## =========== Echo time series data downoad status ============================
@app.callback(
    Output('co-select-echo', 'value'),
    [Input('dl-dataframe', 'children')],
    [State('co-select', 'value')]
)
def retrieve_data(datString, co_name):
    dataDic = json.loads(datString)
    if dataDic['statcode_raw'] == 200:
        return "Data Successfully Downloaded for " + co_name + "  from AlphaVantage Server \n"
    else:
        return "Data could not be downloaded from AlphaVantage Server, Error Code: " + str(code) + "\n"
## =========  PLot raw series =============
@app.callback(
    Output('raw-series', 'figure'),
    [Input('dl-dataframe', 'children'),
     Input('tseries-select-radio', 'value'), 
     Input('ndays-plot-raw', 'value')],
    [State('co-select', 'value')]
)
def update_graph_raw(dataString, tseries_type, ndays_plot, co_select):
    dataDic = json.loads(dataString)
    if dataDic['statcode_raw'] == 200:
        df = pd.read_json(dataDic['df'], orient='split')
        ydata = df[tseries_type]
        xdata = df['date']
        trace1 = go.Scatter(
                x=xdata[0:ndays_plot],
                y=ydata[0:ndays_plot],
                text = ' ',
                mode='lines+markers',
                marker = {
                    "size":15,
                    "opacity":0.5,
                    'color': '#17BECF',
                    "line":{"width":0.5, "color":"white"}
                },
               name =  co_select + " (" + tseries_type.upper() + ")"
             )
        return {
            'data':[trace1],
            'layout':go.Layout(
                xaxis = {'title':' '},
                yaxis = {'title': 'Price (USD)'},
                margin = {'l':60, 'r':40, 't':40, 'b':50},
                hovermode = 'closest',
                title = 'Historical Data'
            )
    
        }
    else:
        pass
### ================== model prediction for fututre ==============================
@app.callback(
    Output('model-run-output', 'children'), 
    [Input('run-model-button', 'n_clicks'),
     Input('dl-dataframe', 'children')],
    [State('ndays-predict', 'value'),
     State('model-select-radio', 'value'),
     State('tseries-select-radio', 'value'),
     State('co-select', 'value'),
     State('ndays-plot-raw', 'value')] 
)
def model_predict(nclicks, dataString, predict_ndays, model_typ, tseries_type, co_sel, n_days_plot): 
    try:
        dataDic = json.loads(dataString)
        df = pd.read_json(dataDic['df'], orient='split')
        print("model_type: ", model_typ)
        if(model_typ =='rnn'):
            tsd = accd.tseries_data(df[tseries_type], df['date'])
            offs = predict_ndays
            series_len = 30
            n_inputs = 1
            n_neurons = 100
            n_outputs = 1
            learning_rate = 0.001
            n_train_iter = 3001

            tf.reset_default_graph()
            xx = tf.placeholder(tf.float64, [None, series_len, n_inputs])
            yy = tf.placeholder(tf.float64, [None, series_len, n_outputs])
            cell = tf.contrib.rnn.OutputProjectionWrapper(
                 tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu),
                 output_size=n_outputs)
            outputs, states = tf.nn.dynamic_rnn(cell, xx, dtype=tf.float64)
            loss = tf.reduce_mean(tf.square(outputs - yy)) 
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
            str_msg = ""
            str_msg = str_msg + "Training Series Length: {0},  Number of Neurons:{1} \n".format(series_len,  n_neurons)
            str_msg = str_msg + "Learning Rate: {0}, Activation Function: {1} \n".format(learning_rate, 'RELU')
            with tf.Session() as sess:
                sess.run(init)            
                for iteration in range(n_train_iter):
                    if iteration == n_train_iter - 1:
                        x_series, y_series = tsd.get_latest_series(series_len, offs)
                    else:
                        x_series, y_series = tsd.get_series(series_len, offs)
                    sess.run(train, feed_dict={xx: x_series, yy: y_series})
                    
                    if iteration % 1000 == 0:            
                        mse = loss.eval(feed_dict={xx: x_series, yy: y_series})
                        str_msg = str_msg + "Iteration: {0},  Mean Square Error: {1:1.5E} \n".format(iteration, mse) 
                        print(str_msg)
                test_x, test_y, test_date = tsd.get_latest_series(series_len, offs, True)    
                y_pred = sess.run(outputs, feed_dict={xx: test_y})


            y_pred = tsd.scaler.inverse_transform(np.reshape(y_pred, [-1,1]))            
            y_pred = y_pred.flatten()
            test_date = test_date.flatten()
            new_date = pd.DataFrame({'date':np.array([pd.to_datetime(test_date[0])+ ii * \
                BDay() for ii in range(1, predict_ndays+1)])[::-1]})
            old_date = pd.DataFrame({'date':test_date[:-predict_ndays*2]})
            model_data_series_date = pd.concat([new_date, old_date])           
            df_model = pd.DataFrame({'date': model_data_series_date['date'],'model_data':y_pred})
            return json.dumps({
                'df': df_model.to_json(orient='split', date_format='iso'),
                'tseries_type' : tseries_type,
                'str_msg' : str_msg,
                'co_select': co_sel                
            })
        elif(model_typ =='arima'):
            model = ARIMA(df[tseries_type].iloc[0:32], order=(1,2,1), dates=df['date'].iloc[0:32])
            model_fit = model.fit(disp=0)
            residuals = pd.DataFrame(model_fit.resid)
            y_pred = model_fit.forecast(steps=3, alpha=0.05) 
            ydata = np.concatenate((y_pred[0][::-1], df[tseries_type].iloc[0:27]), axis=0)  
            old_date = df['date'].iloc[0:30-predict_ndays]
            new_date = pd.DataFrame({'date':np.array([pd.to_datetime(old_date[0])+ ii * \
                BDay() for ii in range(1, predict_ndays+1)])[::-1]})
            old_date = pd.DataFrame({'date':old_date})
            model_data_series_date = pd.concat([new_date, old_date])   
            df_model = pd.DataFrame({'date': model_data_series_date['date'],'model_data':ydata}) 
            str_msg = ""
            str_msg = str_msg + model_fit.summary().as_text()                   
            return json.dumps({
                'df': df_model.to_json(orient='split', date_format='iso'),
                'tseries_type' : tseries_type,
                'str_msg' : str_msg,
                'co_select': co_sel                
            })
        elif(model_typ =='arima_rnn'):
            model = ARIMA(df[tseries_type].iloc[0:], order=(1,2,1), dates=df['date'].iloc[0:])
            model_fit = model.fit(disp=0)
            residuals = pd.DataFrame(model_fit.resid)
            y_pred = model_fit.forecast(steps=3, alpha=0.05) 
 
            old_date = df['date'].iloc[0:27]
            new_date = pd.DataFrame({'date':np.array([pd.to_datetime(old_date[0])+ ii * \
                BDay() for ii in range(1, predict_ndays+1)])[::-1]})
            old_date = pd.DataFrame({'date':old_date})
            model_data_series_date = pd.concat([new_date, old_date])   
            ydata = np.concatenate((y_pred[0][::-1], df[tseries_type].iloc[0:27]), axis=0)        
            df_model = pd.DataFrame({'date': model_data_series_date['date'],'model_data':ydata}) 
            str_msg = ""
            str_msg = str_msg + model_fit.summary().as_text()
            return json.dumps({
                'df': df_model.to_json(orient='split', date_format='iso'),
                'tseries_type' : tseries_type,
                'str_msg' : str_msg,
                'co_select': co_sel                
            })

#        else:
#            return json.dumps({
#                'tseries_type':tseries_type,
#            })
    except ValueError:
        pass
  

## ============== ECHO the model run output ===============================================
@app.callback(
    Output('run-model-echo', 'value'), 
    [Input('run-model-button', 'n_clicks'),
     Input('model-run-output', 'children')]
)
def echo_model_run_output(nclick, datString):
    data_dic = json.loads(datString)   
    try:
        data_dic = json.loads(datString)
        return data_dic['str_msg']
    except ValueError:
        return 'Model Running ....'
## =============== plot technical indicator and raw data ===============================
@app.callback(
    Output('raw-tech-tseries', 'figure'),
    [Input('dl-dataframe', 'children'), 
     Input('tech-ind-list-data', 'children'),     
     Input('tseries-select-radio', 'value'), 
     Input('ndays-plot-raw', 'value') ,
     Input('tech-ind-select', 'value')],
    [State('co-select', 'value')]
)
def update_graph_raw_techInd(dataString, dataString_tech_ind, tseries_type, ndays_plot, \
    tech_ind_typ, co_select):   
    tech_ind_typ_list = []
    tech_ind_typ_list.append(tech_ind_typ)
    print(tech_ind_typ_list)
    data_tech_ind = json.loads(dataString_tech_ind)
    dataDic = json.loads(dataString)
    if dataDic['statcode_raw'] == 200:
        df = pd.read_json(dataDic['df'], orient='split')
        df_tech_ind =  pd.read_json(data_tech_ind['df_techInd'], orient='split')
        ydata = df[tseries_type]
        xdata = df['date']
        trace1 = go.Scatter(
                x=xdata[0:ndays_plot],
                y=ydata[0:ndays_plot],
                text = ' ',
                mode='lines+markers',
                marker = {
                    "size":15,
                    "opacity":0.5,
                    'color': '#17BECF',
                    "line":{"width":0.5, "color":"white"}
                },
               name = co_select + " (" + tseries_type.upper() + ")"
             )
        filtered_df = df_tech_ind.loc[df_tech_ind['property'].isin(tech_ind_typ_list)]
        tech_ind_x = filtered_df['date']        
        tech_ind_y = filtered_df['value']
        trace2 = go.Scatter(
                x=tech_ind_x[0:ndays_plot],
                y=tech_ind_y[0:ndays_plot],
                text = ' ',
                mode='lines+markers',
                marker = {
                    "size":15,
                    "opacity":0.5,
                    'color': '#FF00FF',
                    "line":{"width":0.5, "color":"white"}
                },
                name=tech_ind_typ                
             )

        return {
            'data':[trace2],
            'layout':go.Layout(
                xaxis = {'title':' '},
                yaxis = {'title': 'Price (USD)'},
                margin = {'l':60, 'r':40, 't':40, 'b':50},
                hovermode = 'closest',
                title = 'Technical Indicator',
            )
    
        }
    else:
        pass

    
## =============== plot  raw and modeled data ===============================
@app.callback(
    Output('tech-modeled-tseries', 'figure'),
    [Input('dl-dataframe', 'children'),     
     Input('tseries-select-radio', 'value'), 
     Input('ndays-plot-raw', 'value') ,
     Input('plot-model-button', 'n_clicks')],
    [State('model-run-output', 'children'),
    State('model-select-radio', 'value'),
    State('co-select', 'value')]
)
def update_graph_model_techInd(dataString_raw, tseries_type, ndays_plot, \
    nclicks_plot_pred, dstring_model, model_type, co_select):   
    data_raw = json.loads(dataString_raw)
    df_raw =  pd.read_json(data_raw['df'], orient='split')
    raw_x = df_raw['date']        
    raw_y = df_raw[tseries_type]
    trace1 = go.Scatter(
            x=raw_x[0:ndays_plot],
            y=raw_y[0:ndays_plot],
            mode='lines+markers',
            marker = {
                "size":15,
                "opacity":0.5,
                'color': '#000080',
                "line":{"width":0.5, "color":"white"}
            },
            name= co_select + ' (' +  tseries_type.upper() + ")"
         )

    trace2 = go.Scatter(
            x=raw_x[0:ndays_plot],
            y=raw_y[0:ndays_plot],
            mode='lines+markers',
            marker = {
                "size":15,
                "opacity":0.5,
                'color': '#77b0b1',
                "line":{"width":0.5, "color":"white"}
            },
            name=' '
         )
    if nclicks_plot_pred:
        try:   
            datadic_model = json.loads(dstring_model) 
            print(datadic_model.keys())
            model_tseries_typ = datadic_model['tseries_type']
            dfm = pd.read_json(datadic_model['df'], orient='split')
            ydata_model = dfm['model_data']
            xdata_model = dfm['date']
            co_sel_model = datadic_model['co_select']
            if dfm.shape[0] >  ndays_plot:
                ydata_model = ydata_model[0:ndays_plot]
                xdata_model = xdata_model[0:ndays_plot]
            else:
                ydata_model = ydata_model[:]
                xdata_model = xdata_model[:]

            trace2 = go.Scatter(
                    x=xdata_model,
                    y=ydata_model,
                    text = co_select,
                    mode='lines+markers',
                    marker = {
                        "size":15,
                        "opacity":0.5,
                        'color': '#FF7F50', ##FF7F50
                        "line":{"width":0.5, "color":"white"}
                    },
                    name = model_type.upper() + " (" + co_sel_model + ', ' +  model_tseries_typ.upper() + ")"
                    )
        except ValueError:
            print("Model not ready yet")
    else:
        pass

    return {
        'data':[trace1, trace2],
        'layout':go.Layout(
            xaxis = {'title':' '},
            yaxis = {'title': ' '},
            margin = {'l':40, 'r':10, 't':40, 'b':40},
            hovermode = 'closest',
            title = 'Historical Data & Model Forecast'
    )}  
        


if __name__ == '__main__':
    app.run_server(debug=True)
