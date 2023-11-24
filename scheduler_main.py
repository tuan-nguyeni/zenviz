import dash
from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
from flask import Flask

# Initialize Flask and Dash apps
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Generate time options for dropdown (e.g., 00:00 to 23:30 at 30-minute intervals)
time_options = [{'label': f'{hour:02d}:{minute:02d}', 'value': f'{hour:02d}:{minute:02d}'}
                for hour in range(24) for minute in range(0, 60, 30)]

# Time reference options
time_reference_options = [
    {'label': 'Yesterday', 'value': 'yesterday'},
    {'label': 'Last Week', 'value': 'last_week'},
    {'label': 'Last Month', 'value': 'last_month'},
    {'label': 'Last Year', 'value': 'last_year'}
    # Add more options as needed
]

# App layout
app.layout = dbc.Container([
    html.H3("Data Quality Check Scheduler", className="text-center my-4"),

    # Frequency Selection
    dbc.Row([
        dbc.Col(html.Label("Select Frequency:", className="font-weight-bold"), width=4),
        dbc.Col(
            dcc.Dropdown(
                id='frequency-dropdown',
                options=[
                    {'label': 'Hourly', 'value': 'hourly'},
                    {'label': 'Daily', 'value': 'daily'},
                    {'label': 'Weekly', 'value': 'weekly'}
                ],
                value='daily'
            ),
            width=8
        )
    ], className="mb-3"),

    # Time Selection
    dbc.Row([
        dbc.Col(html.Label("Select Time:", className="font-weight-bold"), width=4),
        dbc.Col(
            dcc.Dropdown(
                id='time-dropdown',
                options=[{'label': f'{hour:02d}:{minute:02d}', 'value': f'{hour:02d}:{minute:02d}'}
                         for hour in range(24) for minute in range(0, 60, 30)],
                value='12:00'
            ),
            width=8
        )
    ], className="mb-3"),

    # Data Created By
    dbc.Row([
        dbc.Col(html.Label("Data Created By:", className="font-weight-bold"), width=4),
        dbc.Col(
            dcc.Dropdown(
                id='time-reference-dropdown',
                options=[
                    {'label': 'Yesterday', 'value': 'yesterday'},
                    {'label': 'Last Week', 'value': 'last_week'},
                    {'label': 'Last Month', 'value': 'last_month'},
                    {'label': 'Last Year', 'value': 'last_year'}
                ],
                value='yesterday'
            ),
            width=8
        )
    ], className="mb-3"),

    # Schedule Button
    dbc.Button("Set Schedule", id='set-schedule-btn', color='primary', className="d-block mx-auto mb-3"),

    # Database Connection Section
    html.H4("Database Connection", className="mt-4 mb-3 text-center"),
    dbc.Row([
        dbc.Col(html.Label("Database Type:", className="font-weight-bold"), width=4),
        dbc.Col(
            dcc.Dropdown(
                id='database-type-dropdown',
                options=[
                    {'label': 'PostgreSQL', 'value': 'postgresql'},
                    {'label': 'Snowflake', 'value': 'snowflake'},
                    {'label': 'Databricks', 'value': 'databricks'}
                ],
                value='postgresql'
            ),
            width=8
        )
    ], className="mb-3"),

    # Connection Details
    dbc.Row([
        dbc.Col(html.Label("Host:", className="font-weight-bold"), width=4),
        dbc.Col(dcc.Input(type="text", placeholder="Host address", className="mb-2"), width=8)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Label("Username:", className="font-weight-bold"), width=4),
        dbc.Col(dcc.Input(type="text", placeholder="Username", className="mb-2"), width=8)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(html.Label("Password:", className="font-weight-bold"), width=4),
        dbc.Col(dcc.Input(type="password", placeholder="Password", className="mb-2"), width=8)
    ], className="mb-3"),

    # Connect Button
    dbc.Button("Connect to Database", id='connect-db-btn', color='primary', className="d-block mx-auto mb-4")
], fluid=True)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
