import dash
from dash import html, dcc, dash_table, Dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
from flask import Flask

# Initialize Flask and Dash apps
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

def process_data():
    # Load valid data from CSV
    df_valid = pd.read_csv("../01_data/german_people_dataset.csv")

    # Manually add invalid data
    df_invalid = pd.DataFrame({
        'Name': ['Emma Müller', 'Lukas Schmidt', 'Niklas Fischer', '', 'Lukas Schmidt'],
        'Age': [39, 200, 200, 73, 200],
        'Yearly Salary': [24180, 98358, 53045, 63115, 96355],
        'City': ['tortmund', 'Essen', 'Hamburg', 'Cologne', 'Berlin'],
        'Job Occupation': ['Engineer', 'Engineer', 'Architect', 'Teacher', 'Journalist'],
        'Issue': ['Wrong city. Corrected to "Dortmund"', 'Age cannot be 200', 'Age cannot be 200', 'Name is missing',
                  'Age cannot be 200']
    })

    return df_valid, df_invalid


df_valid, df_invalid = process_data()

# Dashboard Data (Placeholder values)
year_to_date_error = 30  # in percent
month_to_date_error = 40  # in percent

# App layout
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Valid Data', children=[
            dash_table.DataTable(df_valid.to_dict('records'), [{"name": i, "id": i} for i in df_valid.columns])
        ]),
        dcc.Tab(label='Invalid Data', children=[
            dash_table.DataTable(
                df_invalid.to_dict('records'),
                [{"name": i, "id": i} for i in df_invalid.columns],
                style_data_conditional=[
                    {'if': {'row_index': 'odd'},
                     'backgroundColor': 'rgb(248, 248, 248)'},
                    {'if': {'column_id': 'Issue', 'filter_query': '{Issue} contains "Wrong city"'},
                     'backgroundColor': 'lightgreen',
                     'color': 'black'},
                    {'if': {'column_id': 'Issue',
                            'filter_query': '{Issue} contains "Age cannot" or {Issue} contains "Name is missing"'},
                     'backgroundColor': 'tomato',
                     'color': 'white'}
                ]

            )
        ]),
        dcc.Tab(label='Data Quality Improvement', children=[
            dbc.Container([
                html.H3("Data Quality Improvement Dashboard", className="text-center mb-4"),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            figure=go.Figure(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=year_to_date_error,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Year to Date Error %"}
                                )
                            ),
                            style={'width': '100%', 'height': '300px'}
                        )
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(
                            figure=go.Figure(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=month_to_date_error,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Month to Date Error %"}
                                )
                            ),
                            style={'width': '100%', 'height': '300px'}
                        )
                    ], width=6)
                ])
            ])
        ])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
