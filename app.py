from dash.dependencies import State
import plotly.express as px 
from dash import Dash, dcc, html, Input, Output, callback_context, no_update
import numpy as np

from Models.model import build_model, predict_country
import dash_bootstrap_components as dbc

from random import choice

oModel, lCategories, lNames, oVocab = build_model()

app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.config.suppress_callback_exceptions = True

server = app.server

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    dbc.Col(html.H1("Exercise 5 RNN", style={'text-align': 'left'}),),

    dbc.Col(dbc.Button("Generate Random Name", id='btnGenerate', n_clicks=0, size='lg', style={'text-align': 'left'}),),
    html.Div([
        dcc.Input(
            id='my_txt_input',
            type='text',
            debounce=False,           # changes to input are sent to Dash server only on enter or losing focus
            pattern=r"^[a-zA-Z]+$",  # Regex: string must start with letters only
            spellCheck=False,
            inputMode='latin',       # provides a hint to browser on type of data that might be entered by the user.
            name='text',             # the name of the control, which is submitted with the form data
            #list='browser',          # identifies a list of pre-defined options to suggest to the user
            n_submit=0,              # number of times the Enter key was pressed while the input had focus
            n_submit_timestamp=-1,   # last time that Enter was pressed
            autoFocus=True,          # the element should be automatically focused after the page loaded
            n_blur=0,                # number of times the input lost focus
            n_blur_timestamp=-1,     # last time the input lost focus.
            # selectionDirection='', # the direction in which selection occurred
            # selectionStart='',     # the offset into the element's text content of the first selected character
            # selectionEnd='',       # the offset into the element's text content of the last selected character
        ),
    ]),
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

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

    # ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [
        Output('model_output',          'children'),
        Output("predictions_bar_chart", 'figure'),
        Output("my_txt_input",          'value'),
        Output("vStore_Softmax",        'data'),
        Output("vStore_Raw",            'data'),
    ],

    [
        Input('btnGenerate',    'n_clicks'),
        Input('select_metric',  'value'),
        Input('my_txt_input',   'value'),

        State("vStore_Softmax", 'data'),
        State("vStore_Raw", 'data'),
        
    ]
)
def on_button_press(btnGenerateClicks, select_metric, txtName, stored_sofmax, stored_raw):
    trigger = callback_context.triggered[0]
    print(trigger)

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
        return no_update, fig, no_update, no_update, no_update

    elif trigger['prop_id']=='btnGenerate.n_clicks':
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
        return label, fig, random_name, vSoftmax, vRaw
    elif trigger['prop_id']=='my_txt_input.value':
        print(txtName)
        if txtName and not txtName.isspace() and isEnglish(txtName):
            label, vSoftmax, vRaw  = predict_country(oModel, txtName, oVocab, lCategories)

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
            return label, fig, no_update, vSoftmax, vRaw
    else:
        return no_update, no_update, no_update, no_update

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=False)