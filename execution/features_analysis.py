
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor

# -------------Data---------------------------
df = pd.read_csv("training_set.csv")  #  it is not created yet

X = df[['X', 'V', 'mu', 'T', 'I']] # S, A, V
y = df['qP']

cv = KFold(n_splits=5, shuffle=True, random_state=42)

#------- Wrapper Feature Selection (RFE + Learner) -------------
# ------------------Linear regression------------------------
model = LinearRegression()
selector = RFE(model, n_features_to_select=3)

selector.fit(X, y)

selected_features = X.columns[selector.support_]
print(selected_features)

# ---------------------SVR---------------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rfe', RFE(SVR(kernel='linear'), n_features_to_select=3))
])

pipe.fit(X, y)

# ---------------------CART---------------------------------
tree = DecisionTreeRegressor(random_state=42)
rfe_tree = RFE(tree, n_features_to_select=3)

rfe_tree.fit(X, y)

# ----------------------MLP-----------------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rfe', RFE(LinearRegression(), n_features_to_select=3)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(50,), max_iter=2000))
])

#------- Embedded Feature Selection -------------
# ------------------LASSO-------------------
lasso = LassoCV(cv=5)
lasso.fit(StandardScaler().fit_transform(X), y)

coef = pd.Series(lasso.coef_, index=X.columns)
selected_features = coef[coef != 0].index
print(selected_features)

#---------------------- RF --------------------
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False)
