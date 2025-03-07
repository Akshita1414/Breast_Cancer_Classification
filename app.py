 
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
from dash import dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, precision_recall_curve

# Load dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_frame.iloc[:, :-1] = imputer.fit_transform(data_frame.iloc[:, :-1])

# Scale data
scaler = StandardScaler()
data_frame.iloc[:, :-1] = scaler.fit_transform(data_frame.iloc[:, :-1])

# Split data
X, Y = data_frame.drop(columns='label', axis=1), data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train models
models = {
    'Logistic Regression': LogisticRegression().fit(X_train, Y_train),
    'KNN': KNeighborsClassifier(n_neighbors=5).fit(X_train, Y_train),
    'Random Forest Classifier': RandomForestClassifier(random_state=2).fit(X_train, Y_train)
}

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Breast Cancer Prediction System", className="text-center mt-3"),
    
    dbc.Row([
        dbc.Col([
            html.H5("Enter Tumor Features"),
            *[dbc.Row([
                dbc.Col(html.Label(feature)),
                dbc.Col(dcc.Input(id=f'input-{feature}', type='number', value=0.0, step=0.01))
            ], className="mb-2") for feature in X.columns]
        ], width=4),
        
        dbc.Col([
            html.H5("Select Model"),
            dcc.Dropdown(id="model-selector", options=[{"label": name, "value": name} for name in models.keys()],
                         value="Logistic Regression"),
            html.Br(),
            dbc.Button("Predict", id="predict-btn", color="primary", className="mb-3"),
            html.Div(id="prediction-output", className="alert alert-info")
        ], width=4),
    ]),
    
    html.Hr(),
    html.H3("Model Comparison"),
    dcc.Graph(id="roc-curve"),
    dcc.Graph(id="precision-recall"),
    dcc.Graph(id="feature-distribution")
])

@app.callback(
    [Output("prediction-output", "children"),
     Output("roc-curve", "figure"),
     Output("precision-recall", "figure"),
     Output("feature-distribution", "figure")],
    [Input("predict-btn", "n_clicks")],
    [Input(f'input-{feature}', 'value') for feature in X.columns] + [Input("model-selector", "value")]
)
def update_output(n_clicks, *args):
    if n_clicks is None:
        return dash.no_update
    
    input_values = np.array(args[:-1]).reshape(1, -1)
    model_name = args[-1]
    prediction = models[model_name].predict(input_values)[0]
    prediction_text = "Benign" if prediction == 1 else "Malignant"
    
    # ROC Curve
    fig_roc = px.line(title="ROC Curve")
    for name, model in models.items():
        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, y_scores)
        fig_roc.add_scatter(x=fpr, y=tpr, mode='lines', name=name)
    
    # Precision-Recall Curve
    fig_pr = px.line(title="Precision-Recall Curve")
    for name, model in models.items():
        y_scores = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(Y_test, y_scores)
        fig_pr.add_scatter(x=recall, y=precision, mode='lines', name=name)
    
    # Feature Distribution
    fig_feature = px.histogram(data_frame, x=data_frame.columns[0], color="label", barmode='overlay', title="Feature Distribution")
    
    return f"Prediction: {prediction_text}", fig_roc, fig_pr, fig_feature

server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)

