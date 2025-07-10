from sklearn.ensemble import IsolationForest
import pandas as pd

def detect_anomalies_isolation_forest(df, value_col='radiation', contamination=0.01, random_state=42):
    """
    Detect anomalies in radiation data using Isolation Forest.
    Parameters:
        - df: DataFrame with measurement value column (value_col)
        - contamination: expected anomaly proportion
    Returns:
        - df_out: DataFrame with additional 'anomaly' column (True/False)
    """
    df_out = df.copy()
    X = df_out[[value_col]].values
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    preds = clf.fit_predict(X)
    df_out['anomaly'] = preds == -1
    return df_out 