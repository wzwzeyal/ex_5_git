from dash.dependencies import State
import plotly.express as px 
from dash import Dash, dcc, html, Input, Output, callback_context, no_update
import numpy as np

from Models.model import build_model, predict_country
import dash_bootstrap_components as dbc

from random import choice

oModel, lCategories, lNames, oVocab = build_model()

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

server = app.server

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    dbc.Col(html.H1("Exercise 5 RNN", style={'text-align': 'left'}),),

    dbc.Col(dbc.Button("Generate Random Name", id='button', n_clicks=0, size='lg', style={'text-align': 'left'}),),
    dbc.Col(html.H4(id='model_input', children=[], style={'text-align': 'left'}),),
    dbc.Col(html.H2(id='model_output', children=[], style={'text-align': 'left'}),),
    dbc.Col(    dcc.RadioItems(
                id='select_metric',
                options=[
                    {'label': 'Softmax', 'value': 'Softmax'},
                    {'label': 'Raw', 'value': 'Raw'},
                ],
                value='Softmax'),),

    # # Headine
    # dbc.Row(
    #     [
    #         dbc.Col(html.H1("Exercise 5 RNN", style={'text-align': 'center'}),),
    #     ]
    # ),

    # # Button + Name 
    # dbc.Row(
    #     [
    #         dbc.Col(dbc.Button("Generate Random Name", id='button', n_clicks=0, size='lg', style={'text-align': 'left'}),),
    #         dbc.Col(html.H4(id='model_input', children=[], style={'text-align': 'left'}),),
    #     ],
    #     justify="start"
    # ),

    # html.Br(),

    # # Country
    # dbc.Row(
    #     [
    #         dbc.Col(html.H2(id='model_output', children=[], style={'text-align': 'center'}),),
    #     ]

    # ),

    # # Radio Buttons
    # dbc.Row(
    #     [
    #         dbc.Col(    dcc.RadioItems(
    #             id='select_metric',
    #             options=[
    #                 {'label': 'Softmax', 'value': 'SMAX'},
    #                 {'label': 'Raw', 'value': 'RAW'},
    #             ],
    #             value='SMAX'),),
    #     ]
    # ),

    dbc.Row(
        [
            dbc.Col(dcc.Graph(
                id = 'predictions_bar_chart',
                config={
                      'staticPlot': True,     # True, False
                      'scrollZoom': False,      # True, False
                      'doubleClick': 'reset',  # 'reset', 'autosize' or 'reset+autosize', False
                      'showTips': True,       # True, False
                      'displayModeBar': True,  # True, False, 'hover'
                      'watermark': True,
                      # 'modeBarButtonsToRemove': ['pan2d','select2d'],
                        },
                          ), 
                    ),
        ]
    ),

    dcc.Store(id='vStore_Softmax'),

    dcc.Store(id='vStore_Raw'),
    
])

    # ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [
        Output('model_input',           'children'),
        Output('model_output',          'children'),
        Output("predictions_bar_chart", 'figure'),
        Output("vStore_Softmax",        'data'),
        Output("vStore_Raw",            'data'),
    ],

    [
        Input('button',                 'n_clicks'),
        Input('select_metric',          'value'),

        State("vStore_Softmax", 'data'),
        State("vStore_Raw", 'data'),
    ]
)
def on_button_press(n_clicks, select_metric, stored_sofmax, stored_raw):
    trigger = callback_context.triggered[0]

    if trigger['prop_id']=='select_metric.value':
        if trigger['value']=='Softmax':
            vX = stored_sofmax
        else:
            vX = stored_raw
        vSortedPredictions = sorted(vX)
        vSortedCOls        = sorted(zip(vX, lCategories))
        vY = [col for _,col in vSortedCOls]
        fig = px.bar(x=vSortedPredictions[-5:], y=vY[-5:], orientation='h',)
        fig.update_layout(
            xaxis_title=select_metric,
            yaxis_title="Top 5 Categories",
            font=dict(
                size=24,
                color="RebeccaPurple"))
        return no_update, no_update, fig, no_update, no_update

    elif trigger['prop_id']=='button.n_clicks':
        random_name         = choice(lNames)
        label, vSoftmax, vRaw  = predict_country(oModel, random_name, oVocab, lCategories)

        if select_metric=='Softmax':
            vX = vSoftmax
        else:
            vX = vRaw

        vSortedPredictions = sorted(vX)
        vSortedCOls        = sorted(zip(vX, lCategories))
        vY = [col for _,col in vSortedCOls]

        fig = px.bar(x=vSortedPredictions[-5:], y=vY[-5:], orientation='h',)
        fig.update_layout(
            xaxis_title=select_metric,
            yaxis_title="Top 5 Categories",
            font=dict(
                size=24,
                color="RebeccaPurple")
    )

        return random_name, label, fig, vSoftmax, vRaw
    else:
        return no_update, no_update, no_update, no_update, no_update

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)