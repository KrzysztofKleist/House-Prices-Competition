import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# Define the custom transformer for dictionary mapping
class MappingEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict=None, column=None, handle_unknown="median"):
        self.mapping_dict = mapping_dict if mapping_dict is not None else {}
        self.column = column  # Specify which column to apply the mapping to
        self.handle_unknown = handle_unknown
        # Calculate the median of the provided mapping values
        self.median_value = (
            np.median(list(self.mapping_dict.values())) if self.mapping_dict else 0
        )
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        # Recalculate median in case mapping_dict is updated during fitting
        self.median_value = (
            np.median(list(self.mapping_dict.values())) if self.mapping_dict else 0
        )
        return self

    def transform(self, X):
        # Ensure the column is in the DataFrame and then apply mapping
        X = X.copy()  # Avoid modifying the original DataFrame
        if self.column in X:
            # Apply mapping with the chosen unknown handling strategy
            if self.handle_unknown == "median":
                X[self.column] = (
                    X[self.column]
                    .fillna("Na")
                    .apply(lambda x: self.mapping_dict.get(x, self.median_value))
                )
            elif self.handle_unknown == "zero":
                X[self.column] = (
                    X[self.column]
                    .fillna("Na")
                    .apply(lambda x: self.mapping_dict.get(x, 0))
                )
            else:
                raise ValueError(
                    "Invalid handle_unknown option. Choose 'median' or 'zero'."
                )

            # Standardize the values after mapping
            X[self.column] = self.scaler.fit_transform(
                X[self.column].values.reshape(-1, 1)
            )

        return X


# Custom transformer to map neighborhood values
class NeighborhoodMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict, unknown_cluster="cluster_10"):
        self.mapping_dict = mapping_dict
        self.unknown_cluster = unknown_cluster  # Default cluster for unknown values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_mapped = X.copy()
        if isinstance(X, pd.DataFrame):
            # Map neighborhoods, defaulting to `unknown_cluster` if value is missing in dictionary
            X_mapped["Neighborhood"] = (
                X_mapped["Neighborhood"]
                .map(self.mapping_dict)
                .fillna(self.unknown_cluster)
            )
        return X_mapped


def create_heatmap(df):  # The df should have two columns
    # Create an empty DataFrame
    x = sorted(df.iloc[:, 0].unique().tolist())
    y = sorted(df.iloc[:, 1].unique().tolist())
    empty_data = np.zeros((len(x), len(y)), dtype=int)
    heatmap_df = pd.DataFrame(data=empty_data, index=x, columns=y, dtype=int)
    # Iterate over df
    for index, row in df.iterrows():
        heatmap_df.loc[row[0], row[1]] += 1
    return heatmap_df


def get_variable_name(variable):
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


def get_num_missing_values(df, treshold=0.4):
    missing_val_count_by_column = df.isnull().sum()
    df_name = get_variable_name(df)
    print(f"Missing values in {df_name} columns:")
    missing_val_df = pd.DataFrame(
        missing_val_count_by_column[missing_val_count_by_column > 0]
    ).reset_index()
    missing_val_df.columns = ["attribute", "num_missing"]
    missing_val_df["%_missing"] = (
        round(missing_val_df["num_missing"] / len(df) * 100, 2).astype(str) + "%"
    )
    missing_val_df = missing_val_df.sort_values(
        by="num_missing", ascending=False
    ).reset_index(drop=True)
    print(missing_val_df)

    columns_to_keep = missing_val_df[
        missing_val_df["num_missing"] / len(df) <= treshold
    ]["attribute"].to_list()
    columns_to_drop = missing_val_df[
        missing_val_df["num_missing"] / len(df) > treshold
    ]["attribute"].to_list()

    return missing_val_df  # columns_to_keep, columns_to_drop


def get_categorical_columns(df):
    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))
    print(f"Columns with categorical data:")
    categorical_df = (
        pd.DataFrame(data=d.items(), columns=["attribute", "n_unique"])
        .sort_values(by="n_unique", ascending=False)
        .reset_index(drop=True)
    )
    print(categorical_df)
    return categorical_df


def get_unique_values_column(df, col_name):
    object_cols = [col for col in df.columns if df[col].dtype == "object"]
    if col_name not in object_cols:
        print(f"{col_name} - selected column is not categorical")
        return None
    print(f"{col_name}\t{list(df[col_name].unique())}")
    return None
