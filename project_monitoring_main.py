import dash
from dash import html, dcc, Dash, dash_table
import dash_bootstrap_components as dbc
from flask import Flask
import pandas as pd

# Initialize Flask and Dash apps
server = Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Placeholder data for table metrics
data = {
    "Table": ["User", "Order", "Product"],
    "Entries": [1000, 5000, 1500],
    "Last Data Quality Check": ["2023-09-20", "2023-09-18", "2023-09-19"],
    "Data Quality (%)": [90, 75, 85]
}

# Convert to DataFrame for easier handling in Dash DataTable
df = pd.DataFrame(data)

# Function to generate style data conditional for data quality
def data_quality_style():
    styles = []
    for i in range(len(df)):
        quality = df.loc[i, "Data Quality (%)"]
        color = "green" if quality > 80 else "orange" if quality > 60 else "red"
        styles.append({
            'if': {
                'filter_query': f'{{Table}} eq "{df.loc[i, "Table"]}"',
                'column_id': 'Data Quality (%)'
            },
            'backgroundColor': color,
            'color': 'white'
        })
    return styles

# App layout
app.layout = dbc.Container([
    html.H3("Database Tables Data Quality Monitoring", className="text-center my-4"),
    dash_table.DataTable(
        df.to_dict('records'),
        [{"name": i, "id": i} for i in df.columns],
        style_data_conditional=data_quality_style()
    )
], fluid=True)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
