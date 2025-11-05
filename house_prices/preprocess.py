import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import mlflow



def split_data(df, cont_features, cat_nom_features, cat_ord_features, label_col, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    cont_features : list
        List of continuous feature names.
    cat_nom_features : list
        List of categorical nominal feature names.
    cat_ord_features : list
        List of categorical ordinal feature names.
    label_col : str
        Name of the target column.
    test_size : float
        Proportion of dataset to include in the test split.
    random_state : int
        Random state for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.Series
        Split training and testing data.
    """
    
    X = df[cont_features + cat_nom_features + cat_ord_features].copy()
    y = df[label_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def scale_continuous_features(X_train, X_test, cont_features):
    """
    Scales continuous (numerical) features using StandardScaler.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Testing feature set.
    cont_features : list
        List of continuous feature names.

    Returns
    -------
    X_train_scaled : np.ndarray
        Scaled training continuous features.
    X_test_scaled : np.ndarray
        Scaled testing continuous features.
    scaler : StandardScaler
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[cont_features])
   
    

    X_train_scaled = scaler.transform(X_train[cont_features])
    X_test_scaled  = scaler.transform(X_test[cont_features])
    joblib.dump(scaler, "/Users/ebotfabien/Desktop/school/hosuing_pipelinw/dsp-fabienmbi-ebot/models/standard_scaler.joblib", compress=3) #why compress 3
    mlflow.log_artifact("/Users/ebotfabien/Desktop/school/hosuing_pipelinw/dsp-fabienmbi-ebot/models/standard_scaler.joblib")  # log to MLflow

    return X_train_scaled, X_test_scaled, scaler

def encode_nominal_features(X_train, X_test, cat_nom_features):
    """
    Encodes nominal (non-ordinal categorical) features using OneHotEncoder.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Testing feature set.
    cat_nom_features : list
        List of nominal categorical feature names.

    Returns
    -------
    X_train_encoded : np.ndarray
        One-hot encoded training features.
    X_test_encoded : np.ndarray
        One-hot encoded testing features.
    encoder : OneHotEncoder
        Fitted OneHotEncoder instance.
    ohe_cols : list
        List of new column names for the one-hot encoded features.
    """
    # Initialize and fit the encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[cat_nom_features])  # fit only on training data


    # Transform datasets
    X_train_encoded = encoder.transform(X_train[cat_nom_features])
    X_test_encoded  = encoder.transform(X_test[cat_nom_features])

    # Get new column names
    ohe_cols = list(encoder.get_feature_names_out(cat_nom_features))
    joblib.dump(encoder, "/Users/ebotfabien/Desktop/school/hosuing_pipelinw/dsp-fabienmbi-ebot/models/one_hot_encoder.joblib", compress=3)
    mlflow.log_artifact("/Users/ebotfabien/Desktop/school/hosuing_pipelinw/dsp-fabienmbi-ebot/models/one_hot_encoder.joblib")
    return X_train_encoded, X_test_encoded, encoder, ohe_cols

def encode_ordinal_features(X_train, X_test, cat_ord_features, ord_categories=None):
    """
    Encodes ordinal categorical features using OrdinalEncoder.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Testing feature set.
    cat_ord_features : list
        List of ordinal categorical feature names.
    ord_categories : list of lists, optional
        Predefined order of categories for each ordinal feature.

    Returns
    -------
    X_train_encoded : np.ndarray
        Ordinally encoded training features.
    X_test_encoded : np.ndarray
        Ordinally encoded testing features.
    encoder : OrdinalEncoder
        Fitted OrdinalEncoder instance.
    """
    encoder = OrdinalEncoder(
        categories=ord_categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # Fit only on training data
    encoder.fit(X_train[cat_ord_features])

    # Transform both train and test sets
    X_train_encoded = encoder.transform(X_train[cat_ord_features])
    X_test_encoded  = encoder.transform(X_test[cat_ord_features])
    joblib.dump(encoder, "/Users/ebotfabien/Desktop/school/hosuing_pipelinw/dsp-fabienmbi-ebot/models/ordinal_encoder.joblib", compress=3)
    mlflow.log_artifact("/Users/ebotfabien/Desktop/school/hosuing_pipelinw/dsp-fabienmbi-ebot/models/ordinal_encoder.joblib")

    return X_train_encoded, X_test_encoded, encoder

def create_processed_dfs(X_train, X_test,
                         X_train_cont, X_test_cont, cont_features,
                         X_train_ohe, X_test_ohe, ohe_cols,
                         X_train_ord, X_test_ord, cat_ord_features):
    """
    Reconstructs DataFrames for scaled/encoded features with appropriate column names and indices.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Original training and test feature DataFrames (for index reference).
    X_train_cont, X_test_cont : np.ndarray
        Scaled continuous feature arrays.
    cont_features : list
        List of continuous feature names.
    X_train_ohe, X_test_ohe : np.ndarray
        One-hot encoded feature arrays.
    ohe_cols : list
        List of one-hot encoded feature names.
    X_train_ord, X_test_ord : np.ndarray
        Ordinally encoded feature arrays.
    cat_ord_features : list
        List of ordinal categorical feature names.

    Returns
    -------
    df_train_final, df_test_final : pd.DataFrame
        Combined preprocessed training and testing DataFrames.
    """
    # Continuous features DataFrame
    df_train_cont = pd.DataFrame(X_train_cont, index=X_train.index, columns=cont_features)
    df_test_cont  = pd.DataFrame(X_test_cont,  index=X_test.index,  columns=cont_features)

    # One-Hot encoded features DataFrame
    df_train_ohe = pd.DataFrame(X_train_ohe, index=X_train.index, columns=ohe_cols)
    df_test_ohe  = pd.DataFrame(X_test_ohe,  index=X_test.index,  columns=ohe_cols)

    # Ordinal encoded features DataFrame
    df_train_ord = pd.DataFrame(X_train_ord, index=X_train.index, columns=cat_ord_features)
    df_test_ord  = pd.DataFrame(X_test_ord,  index=X_test.index,  columns=cat_ord_features)

    # Combine all into single DataFrames
    df_train_final = pd.concat([df_train_cont, df_train_ohe, df_train_ord], axis=1)
    df_test_final  = pd.concat([df_test_cont, df_test_ohe, df_test_ord], axis=1)

    return df_train_final, df_test_final

