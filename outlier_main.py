import base64
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, dash
from flask import Flask
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px
import logging
import ydata_profiling

from sklearn.impute import SimpleImputer

import dash_bootstrap_components as dbc
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'])
app.title = 'ZenViz'  # Set the title here
app.index_string = open('custom_index_string.html').read()
server = app.server  # This is important for Gunicorn


#download_button = html.Button("Download CSV", id="btn-download-csv")
#download_link = dcc.Download(id="download-dataframe-csv")

# App layout
app.layout = html.Div([
    # Header
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("About Us", href="https://clean-your-data.com/")),
            dbc.NavItem(dbc.NavLink("Contact Us",
                                    href="https://forms.monday.com/forms/52f24ae10b0aa29b110b0dc2e1d37af1?r=euc1"))
        ],
        brand="ZenViz",
        brand_href="#",
        color="primary",
        dark=True,
    ),

    # Title and Description
    html.H1("Visualizing Data Errors in Seconds", style={'textAlign': 'center', 'marginTop': '20px'}),
    html.P(
        "Understand your data and the corresponding data quality by uploading your CSV file. More formats will follow soon.",
        style={'textAlign': 'center', 'marginBottom': '20px'}
    ),

    # File Upload Section
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.I(className="fa fa-upload", style={'fontSize': '24px', 'marginRight': '10px'}),
            'Drop CSV ',
            html.A('here')
        ], style={'fontSize': '20px'}),
        style={
            'width': '100%',
            'height': '200px',
            'lineHeight': '200px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '10px',
            'color': '#007bff',
            'borderColor': '#007bff'
        },
        multiple=False
    ),

    # Additional Description
    html.P(
        "Choose what you want to see by clicking on it",
        style={'textAlign': 'center', 'marginTop': '10px', 'marginBottom': '20px'}
    ),
    #download_button,
    #download_link,
    dcc.Tabs([
        dcc.Tab(label='See the full dataset', children=[
            dash_table.DataTable(
                id='full-data-table',
                editable=True,  # Enable editing
            )

        ]),
        dcc.Tab(label='Data Profile', children=[
            dcc.Loading(
                id="loading-profile",
                children=[html.Iframe(id='profile-report-container', style={"width": "100%", "height": "100vh"})],
                type="default",  # You can use "circle", "cube", "dot", or "default"
            )
        ])

        ,

        dcc.Tab(label='See missing values', children=[
            html.Div(id='missing-data-container')
        ]),
        dcc.Tab(label='See outliers', children=[
            html.Div(id='outlier-data-container'),
            dcc.Dropdown(
                id='isolation-method-dropdown',
                value='all'
            ),
            #html.Div(id='isolation-outlier-data-container')
        ]),
        dcc.Tab(label='See duplicates', children=[
            html.Div([

                html.Div(id='duplicates-output')
            ])
        ]),
        dcc.Tab(label='Data Visualization', children=[
            dcc.Dropdown(id='analysis-column-dropdown'),
            html.Div(id='column-analysis-container')
        ])
    ])
])
# ... [previous code] ...

@app.callback(
    Output('duplicates-output', 'children'),
    Input('upload-data', 'contents')
)
def auto_detect_duplicates(contents):
    if contents is None:
        return 'Upload a file to detect duplicates.'

    df = parse_contents(contents)
    if df is None:
        return 'Error in processing file.'

    # Detecting duplicates
    duplicate_rows = df[df.duplicated(keep=False)]
    if duplicate_rows.empty:
        return 'No duplicates found.'

    return html.Div([
        html.H4('Duplicate Rows'),
        dash_table.DataTable(
            duplicate_rows.to_dict('records'),
            [{"name": i, "id": i} for i in duplicate_rows.columns],
            style_table={'overflowX': 'scroll'}
        )
    ])

# ... [rest of the code] ...


@app.callback(
    Output('missing-data-container', 'children'),
    Input('upload-data', 'contents')
)
def display_missing_data(contents):
    if contents is not None:
        logging.info('display_missing_data called')
        df = parse_contents(contents)
        if df is not None:
            # Check for missing data
            missing_data_df = df[df.isnull().any(axis=1)]
            return generate_missing_data_table(missing_data_df)
    return 'Please upload data to view missing data.'

def generate_missing_data_table(df):
    """Generate a table with highlighted cells for missing data."""
    style_data_conditional = []
    for col in df.columns:
        style = {
            'if': {
                'filter_query': f'{{{col}}} is blank',
                'column_id': col
            },
            'backgroundColor': '#FF4136',  # Highlight color
            'color': 'white'
        }
        style_data_conditional.append(style)

    return dash_table.DataTable(
        df.to_dict('records'),
        [{"name": i, "id": i} for i in df.columns],
        style_data_conditional=style_data_conditional
    )

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in content_type:
            # Set chunksize to a suitable value
            chunksize = 10000  # Example chunksize, adjust based on your needs

            # Initialize an empty DataFrame for the chunks
            full_data = pd.DataFrame()

            for chunk in pd.read_csv(io.StringIO(decoded.decode('utf-8')), chunksize=chunksize):
                # Optimize data types
                for col in chunk.columns:
                    if chunk[col].dtype == 'object':
                        num_unique_values = len(chunk[col].unique())
                        num_total_values = len(chunk[col])
                        if num_unique_values / num_total_values < 0.5:
                            chunk[col] = chunk[col].astype('category')

                # Append chunk to full data
                full_data = pd.concat([full_data, chunk])

            return full_data

        else:
            return html.Div(['File format not supported.'])
    except Exception as e:
        return html.Div(['There was an error processing this file.'])


@app.callback(
    [Output('full-data-table', 'data'),
     Output('full-data-table', 'columns'),  # Add this line to update the columns
     Output('analysis-column-dropdown', 'options')],
    Input('upload-data', 'contents')
)
def display_full_data(contents):
    if contents is not None:
        logging.info('display_full_data called')
        df = parse_contents(contents)
        columns = [{"name": i, "id": i} for i in df.columns]  # Define columns for DataTable
        options = [{'label': col, 'value': col} for col in df.columns]

        return df.to_dict('records'), columns, options

    return [], [], []  # Return empty lists if no contents


@app.callback(
    Output('outlier-data-container', 'children'),
    Input('upload-data', 'contents')
)
def display_outliers(contents):
    if contents is not None:
        df = parse_contents(contents)
        if df is not None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            outlier_dfs = []
            for col in numeric_cols:
                outliers = standard_deviation_outlier(df, col)
                if outliers is not None and not outliers.empty:
                    outlier_dfs.append(html.Div([
                        html.H4(f'Outliers in {col}'),
                        dash_table.DataTable(outliers.to_dict('records'), [{"name": i, "id": i} for i in outliers.columns])
                    ]))
            return html.Div(outlier_dfs)
    return 'Please upload data to detect outliers.'

@app.callback(
    Output('column-analysis-container', 'children'),
    [Input('analysis-column-dropdown', 'value'),
     Input('upload-data', 'contents')]
)
def update_column_analysis(column_name, contents):
    if contents is not None and column_name is not None:
        logging.info(f"column: {column_name} called")
        df = parse_contents(contents)
        return column_analysis(df, column_name)
    return 'Select a column for analysis.'


def is_numeric(col):
    return pd.api.types.is_numeric_dtype(col)

def standard_deviation_outlier(df, col):
    if is_numeric(df[col]):
        deviation = np.std(df[col])
        mean_val = np.mean(df[col])
        threshold = 3 * deviation
        outliers = df[np.abs(df[col] - mean_val) > threshold]
        return outliers
    else:
        return pd.DataFrame()

def column_analysis(df, col):
    missing_values = df[col].isnull().sum()
    data_type = 'Categorical' if df[col].dtype == 'object' else 'Numeric'
    unique_values = df[col].nunique() if data_type == 'Categorical' else 'N/A'
    analysis_info = [
        html.H4(f'Analysis of Column: {col}'),
        html.P(f'Missing Values: {missing_values}'),
        html.P(f'Data Type: {data_type}'),
        html.P(f'Unique Values: {unique_values}')
    ]

    if data_type == 'Numeric':
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[col], name=col))
        fig.update_layout(title_text="Box Plot")
        analysis_info.append(dcc.Graph(figure=fig))

    return html.Div(analysis_info)

def generate_profile_report(df):
    # Generate the report
    profile = ydata_profiling.ProfileReport(df)
    # Convert the report to HTML
    report_html = profile.to_html()
    return report_html

@app.callback(
    Output('profile-report-container', 'srcDoc'),
    Input('upload-data', 'contents')
)
def update_profile_report(contents):
    if contents is not None:
        df = parse_contents(contents)
        if df is not None:
            report_html = generate_profile_report(df)
            return report_html
    # Instead of returning a static message, return None to keep the loading spinner active
    # until the report is generated.
    return None



if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
