import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
print(linear_model.score(three_var, recipes['calories']))
print(linear_model.coef_)
print(linear_model.intercept_)

tag_dict = dict()

tags= recipes['tags'].str.strip("[]")
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
## removing all tags that are keys for something else
tag_dummies = tags.str.get_dummies(sep=",")
tag_dummies = tag_dummies[[' \'easy\'', ' \'dietary\'', ' \'main-dish\'', ' \'low-in-something\'', ' \'meat\'', ' \'vegetables\'']]

recipes['easy'] = tag_dummies[' \'easy\'']
recipes['dietary'] = tag_dummies[' \'dietary\'']
recipes['main-dish'] = tag_dummies[' \'main-dish\'']
recipes['low-in-something'] = tag_dummies[' \'low-in-something\'']
recipes['meat'] = tag_dummies[' \'meat\'']
recipes['vegetables'] = tag_dummies[' \'vegetables\'']

nine_var = recipes[['minutes','n_steps', 'n_ingredients', 'easy', 'dietary', 'main-dish', 'low-in-something', 'meat', 'vegetables']]

print("Calories Model")
linear_with_dummy_vars = LinearRegression().fit(nine_var, recipes['calories'])
print(linear_with_dummy_vars.score(nine_var, recipes['calories']))
print(linear_with_dummy_vars.coef_)
print(linear_with_dummy_vars.intercept_)

print("\n=========== SUMMARY ===========")
xlabels = nine_var.columns.values
stats.summary(linear_with_dummy_vars, nine_var, recipes['calories'], xlabels)

print("Protein Model")
ldv_protein = LinearRegression().fit(nine_var, recipes['protein'])
print(ldv_protein.score(nine_var, recipes['protein']))
print(ldv_protein.coef_)
print(ldv_protein.intercept_)

print("\n=========== SUMMARY ===========")
stats.summary(linear_with_dummy_vars, nine_var, recipes['protein'], xlabels)

print("Total Fat Model")
ldv_tfat = LinearRegression().fit(nine_var, recipes['total_fat'])
print(ldv_tfat.score(nine_var, recipes['total_fat']))
print(ldv_tfat.coef_)
print(ldv_tfat.intercept_)

print("\n=========== SUMMARY ===========")
stats.summary(linear_with_dummy_vars, nine_var, recipes['total_fat'], xlabels)

print("Carbs Model")
ldv_carbs = LinearRegression().fit(nine_var, recipes['carbs'])
print(ldv_carbs.score(nine_var, recipes['carbs']))
print(ldv_carbs.coef_)
print(ldv_carbs.intercept_)

print("\n=========== SUMMARY ===========")
stats.summary(linear_with_dummy_vars, nine_var, recipes['carbs'], xlabels)

## Ridge regression for the protein model
