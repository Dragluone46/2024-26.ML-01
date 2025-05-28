import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    make_scorer,
)
import sklearn

sklearn.set_config(transform_output="pandas")

df = pd.read_csv(r"C:\Users\ChristianDiCeglie\Documents\prova\ML\ml_datasets\Salary Data.csv")

df.dropna(axis=0, inplace=True)

x = df.drop(columns=["Salary"])
y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

encoder = ColumnTransformer(
    [
        (
            "onehot",
            OneHotEncoder(
                sparse_output=False, min_frequency=5, handle_unknown="infrequent_if_exist"
            ),
            ["Gender", "Job Title", "Education Level"],
        )
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)

pipe = Pipeline(
    [
        ("encoder", encoder),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42)),
    ]
)
params = {
    'model__n_estimators': [100, 200, 300],
    'model__criterion': ['squared_error', 'absolute_error'],
    'encoder__onehot__min_frequency': [1, 4, 7]
    }

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    n_jobs=-1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    refit=True,
    verbose=4
)

grid_search.fit(x_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

pipe = pipe.set_params(**grid_search.best_params_)
pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test)
print("MAE: ", mean_absolute_error(y_test, y_pred))