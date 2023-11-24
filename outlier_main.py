import base64
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
from flask import Flask
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px

# Initialize Flask and Dash apps
from sklearn.impute import SimpleImputer

# Initialize Dash app
app = Dash(__name__)
server = app.server  # This is important for Gunicorn

# App layout
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    dcc.Tabs([
        dcc.Tab(label='Full Data', children=[
            html.Div(id='full-data-table')
        ]),
        dcc.Tab(label='Missing Data', children=[
            html.Div(id='missing-data-container')
        ]),
        dcc.Tab(label='Outliers', children=[
            html.Div(id='outlier-data-container'),
            dcc.Dropdown(
                id='isolation-method-dropdown',
                options=[
                    {'label': 'All Columns', 'value': 'all'},
                    {'label': 'Numeric Columns Only', 'value': 'numeric'}
                ],
                value='all'
            ),
            html.Div(id='isolation-outlier-data-container')
        ]),
        dcc.Tab(label='Column Analysis', children=[
            dcc.Dropdown(id='analysis-column-dropdown'),
            html.Div(id='column-analysis-container')
        ])
    ])
])

@app.callback(
    Output('missing-data-container', 'children'),
    Input('upload-data', 'contents')
)
def display_missing_data(contents):
    if contents is not None:
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


@app.callback(
    Output('isolation-outlier-data-container', 'children'),
    [Input('isolation-method-dropdown', 'value'),
     Input('upload-data', 'contents')]
)
def display_isolation_forest_outliers(method, contents):
    if contents is not None:
        df = parse_contents(contents)
        if df is not None:
            if method == 'all':
                # Apply Isolation Forest to all columns
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                return isolation_forest_outliers(df, numeric_cols)
            elif method == 'numeric':
                # Apply Isolation Forest to numeric columns only
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                return isolation_forest_outliers(df, numeric_cols)
    return 'Please upload data and select a method.'

def isolation_forest_outliers(df, cols):
    # Impute missing values with median for numeric columns
    imputer = SimpleImputer(strategy='median')
    df_imputed = df.copy()
    df_imputed[cols] = imputer.fit_transform(df[cols])

    # Apply Isolation Forest
    clf = IsolationForest(max_samples=100, random_state=1, contamination='auto')
    df_imputed['Outlier'] = clf.fit_predict(df_imputed[cols])

    # Filter out the outliers
    outliers_df = df_imputed[df_imputed['Outlier'] == -1]

    # Apply PCA to reduce to two dimensions
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_imputed[cols])
    df_imputed[['PC1', 'PC2']] = principal_components

    # Plot with Plotly
    fig = px.scatter(df_imputed, x='PC1', y='PC2', color='Outlier',
                     title="PCA - 2D Scatter Plot of Data with Outliers Highlighted",
                     labels={'Outlier': 'Outlier (-1 for Outlier, 1 for Inlier)'})

    # Generate a table of outliers
    outliers_table = dash_table.DataTable(
        outliers_df.to_dict('records'),
        [{"name": i, "id": i} for i in outliers_df.columns],
        style_table={'overflowX': 'scroll'}
    )

    return html.Div([
        html.H4('Isolation Forest Outliers'),
        html.Div(outliers_table),
        html.H4('Isolation Forest Outliers with PCA Visualization'),
        dcc.Graph(figure=fig)
    ])


def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in content_type:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.Div(['File format not supported.'])
        return df
    except Exception as e:
        return html.Div(['There was an error processing this file.'])

@app.callback(
    [Output('full-data-table', 'children'),
     Output('analysis-column-dropdown', 'options')],
    Input('upload-data', 'contents')
)
def display_full_data(contents):
    if contents is not None:
        df = parse_contents(contents)
        options = [{'label': col, 'value': col} for col in df.columns]
        return dash_table.DataTable(
            df.to_dict('records'),
            [{"name": i, "id": i} for i in df.columns],
            style_table={'overflowX': 'scroll'}
        ), options
    return 'Please upload a file to see the data.', []

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

if __name__ == '__main__':
    app.run_server(debug=True)
