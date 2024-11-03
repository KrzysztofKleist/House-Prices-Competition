# House-Prices-Competition

## My submission for Kaggle House Prices Competition.

Full solution is available [here](/house-competition-model-stacking.ipynb).

### Preprocessing

The notebook shows all the preprocessing steps, where I built a custom `MappingEncoder` for categorical ordinal variables.
Then I also reduced the number of attributes for Neighborhood column (from 25 unique values to 9) by implementing KMeans and then mapped using `NeigborhoodMapper`.
The rest of the categorical data was one hot encoded.

### Model

For the model I used base models `XGBRegressor` and `LGBMRegressor` with meta-model `Ridge`, they all worked in a `StackingRegressor`.
Everything was enclosed in a `Pipeline` with a `ColumnTransformer`.
The model parameters were optimized useing `GridSearchCV`. Unfortunately the search was limited due to lack of computational resources.

### Results

Without extensive research of parameters my model scored 14870.8 what places me in top 5% of all the results!
