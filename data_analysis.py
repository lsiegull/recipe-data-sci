import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy import stats
from regressors import stats

recipes = pd.read_csv("RAW_recipes.csv")

## nutrition array is calories, total_fat, sugar, sodium, protein, sat_fat, carbs

nutrition = recipes['nutrition'].str.strip("[]").str.split(",")

nutri_dataframe = pd.DataFrame.from_records(nutrition)

recipes['calories'] = nutri_dataframe[0].astype(np.float64)
recipes['total_fat'] = nutri_dataframe[1].astype(np.float64)
recipes['sugar'] = nutri_dataframe[2].astype(np.float64)
recipes['sodium'] = nutri_dataframe[3].astype(np.float64)
recipes['protein'] = nutri_dataframe[4].astype(np.float64)
recipes['sat_fat'] = nutri_dataframe[5].astype(np.float64)
recipes['carbs'] = nutri_dataframe[6].astype(np.float64)

recipes['minutes'] = recipes['minutes'].astype(np.float64)
recipes['n_steps'] = recipes['n_steps'].astype(np.float64)
recipes['n_ingredients'] = recipes['n_ingredients'].astype(np.float64)

three_var = recipes[['minutes','n_steps', 'n_ingredients']]

linear_model = LinearRegression().fit(three_var, recipes['calories'])

xlabels = three_var.columns.values
print("\n=========== SUMMARY for Three Var Calories Linear Regression ===========")
stats.summary(linear_model, three_var, recipes['calories'], xlabels)

tag_dict = dict()

tags= recipes['tags'].str.strip("[]")

## find top tags & remove ones that are keys for something else/aren't useful
"""
for row in tags:
    tag_str = row.split(",")
    for t in tag_str:
        if t in tag_dict:
            tag_dict[t] = tag_dict[t] + 1
        else:
            tag_dict[t] = 1
tag_vals = list(tag_dict.values())
tag_vals.sort(reverse=True)

top_five = tag_vals[0:15]
for key in tag_dict.keys():
    if tag_dict[key] in top_five:
        print(key)
        print(tag_dict[key])
"""

tag_dummies = tags.str.get_dummies(sep=",")
tag_dummies = tag_dummies[[' \'easy\'', ' \'dietary\'', ' \'main-dish\'', ' \'low-in-something\'', ' \'meat\'', ' \'vegetables\'']]

recipes['easy'] = tag_dummies[' \'easy\'']
recipes['dietary'] = tag_dummies[' \'dietary\'']
recipes['main-dish'] = tag_dummies[' \'main-dish\'']
recipes['low-in-something'] = tag_dummies[' \'low-in-something\'']
recipes['meat'] = tag_dummies[' \'meat\'']
recipes['vegetables'] = tag_dummies[' \'vegetables\'']

nine_var = recipes[['minutes','n_steps', 'n_ingredients', 'easy', 'dietary', 'main-dish', 'low-in-something', 'meat', 'vegetables']]

linear_with_dummy_vars = LinearRegression().fit(nine_var, recipes['calories'])

print("\n=========== SUMMARY for Nine Var Calories Linear Regression ===========")
xlabels = nine_var.columns.values
stats.summary(linear_with_dummy_vars, nine_var, recipes['calories'], xlabels)

ldv_protein = LinearRegression().fit(nine_var, recipes['protein'])

print("\n=========== SUMMARY for Protein Linear Regression ===========")
stats.summary(ldv_protein, nine_var, recipes['protein'], xlabels)

ldv_tfat = LinearRegression().fit(nine_var, recipes['total_fat'])

print("\n=========== SUMMARY for Total Fat Linear Regression ===========")
stats.summary(ldv_tfat, nine_var, recipes['total_fat'], xlabels)

ldv_carbs = LinearRegression().fit(nine_var, recipes['carbs'])

print("\n=========== SUMMARY for Carbs Linear Regression ===========")
stats.summary(ldv_carbs, nine_var, recipes['carbs'], xlabels)

## Ridge regression for the protein model

n_samples, n_features = 2000, 9
rng = np.random.RandomState(0)
y_data = rng.randn(n_samples)
x_data = rng.randn(n_samples, n_features)
rr_protein = Ridge(alpha = 1.0)
rr_protein.fit(nine_var, recipes['protein'])
rr_protein.score(nine_var, recipes['protein'])

print("\n=========== SUMMARY for Protein Ridge Regression ===========")
stats.summary(rr_protein, nine_var, recipes['protein'], xlabels)

## PCA for the protein model
print("\n=========== PCA for Protein Linear Model ===========")
pca = PCA(n_components = 3)
pca.fit(nine_var)
print("Explained Variance Ratio")
print(pca.explained_variance_ratio_)
print("Singular Values")
print(pca.singular_values_)
components = pca.fit_transform(nine_var)
pca_data = pd.DataFrame(data= components, columns=['PC1', 'PC2', 'PC3'])

xlabels = pca_data.columns.values
pca_protein = LinearRegression().fit(pca_data, recipes['protein'])
print("\n=========== SUMMARY for PCA Protein Linear Regression ===========")
stats.summary(pca_protein, pca_data, recipes['protein'], xlabels)
