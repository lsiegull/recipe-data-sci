import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

#tag_dummies = recipes['tags'].str.strip("[]").str.get_dummies(sep=",")
#print(tag_dummies)
#print(type(tag_dummies))
