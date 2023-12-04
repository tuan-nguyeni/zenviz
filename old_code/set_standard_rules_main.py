import dash
from dash import html, dcc, Dash
import dash_bootstrap_components as dbc
from flask import Flask

# Initialize Flask and Dash apps
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample dropdown options for column names (modify as needed)
column_options = [
    {'label': 'Age', 'value': 'age'},
    {'label': 'Salary', 'value': 'salary'},
    {'label': 'City', 'value': 'city'},
    # Add more options as required
]

# Function to add a new rule input pair with dropdown
def add_rule_input_pair(n_clicks):
    return html.Div([
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id=f'column-dropdown-{n_clicks}',  # Unique ID for each dropdown
                    options=column_options,
                    placeholder="Select Column",
                ),
                width=4
            ),
            dbc.Col(
                dcc.Input(
                    placeholder="Rule (e.g., 'age is between 15-90')",
                    type="text",
                    className="mb-2",
                    style={"width": "100%"}
                ),
                width=8
            ),
        ], className="mt-2")
    ])

# App layout
app.layout = dbc.Container([
    html.H3("Set Predefined Rules", className="mb-4 mt-4"),

    # Checkboxes for predefined rules
    dbc.Row([
        dbc.Col(dbc.Checkbox(id='outlier-detection-checkbox', className="form-check-input"), width=1),
        dbc.Col(html.Label("Outlier Detection", htmlFor='outlier-detection-checkbox', className="form-check-label"),
                width=11),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dbc.Checkbox(id='spelling-mistakes-checkbox', className="form-check-input"), width=1),
        dbc.Col(html.Label("Spelling Mistakes", htmlFor='spelling-mistakes-checkbox', className="form-check-label"),
                width=11),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dbc.Checkbox(id='address-correction-checkbox', className="form-check-input"), width=1),
        dbc.Col(html.Label("Delete Duplicates", htmlFor='address-correction-checkbox', className="form-check-label"),
                width=11),
    ], className="mb-3"),

    # Section for custom rules
    html.H4("Custom Rules", className="mb-3"),
    html.Div(id='custom-rules-container', children=[]),
    html.Button('Add Rule', id='add-rule-button', n_clicks=0, className="mt-2 btn btn-primary"),
], fluid=True)

# Callback to add new rule input pair
@app.callback(
    dash.dependencies.Output('custom-rules-container', 'children'),
    [dash.dependencies.Input('add-rule-button', 'n_clicks')],
    [dash.dependencies.State('custom-rules-container', 'children')]
)
def display_custom_rules(n_clicks, children):
    new_child = add_rule_input_pair(n_clicks)
    children.append(new_child)
    return children

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
