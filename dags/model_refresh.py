from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from pendulum import datetime
import pandas as pd
from airflow.operators.empty import EmptyOperator


@dag(
    start_date=datetime(2024, 8, 1),
    schedule="@monthly",
    catchup=False,
)
def model_refresh():

    @task
    def extract_latest_data(**context):
        from include.utils import generate_churn_data

        today = context["logical_date"]

        new_data = generate_churn_data(num_customers=20, base_date=today)

        return new_data

    @task
    def get_historical_data(filepath="include/churn_data.csv"):
        historical_data = pd.read_csv(filepath)

        return historical_data

    @task
    def combine_data(new_data, historical_data):
        combined_data = pd.concat([new_data, historical_data])

        combined_data = pd.DataFrame(combined_data).drop_duplicates()

        return combined_data

    @task
    def load_new_data_to_db(new_data):

        new_data.to_csv("churn_data.csv", mode="a", header=False, index=False)

    @task
    def feature_engineering(data, **context):
        import pandas as pd
        import numpy as np

        base_date = context["logical_date"]

        data["signup_date"] = pd.to_datetime(data["signup_date"], utc=True)
        data["churn_date"] = pd.to_datetime(data["churn_date"], utc=True)
        base_date = pd.to_datetime(base_date, utc=True)

        data["account_age_days"] = (base_date - data["signup_date"]).dt.days

        data["support_interaction_rate"] = data["support_calls"] / data["tenure"]
        data["high_support_individual"] = np.where(
            (data["tier"] == "Individual") & (data["support_calls"] > 5), 1, 0
        )
        data["product_tier_combined"] = data["product"] + " - " + data["tier"]

        data["monthly_charge_bucket"] = pd.cut(
            data["monthly_charges"],
            bins=[0, 50, 100, np.inf],
            labels=["Low", "Medium", "High"],
        )
        data["total_spend"] = data["monthly_charges"] * data["tenure"]

        data = pd.get_dummies(
            data, columns=["operating_system"], prefix="os", drop_first=True
        )
        data = pd.get_dummies(data, columns=["tier"], prefix="tier", drop_first=True)

        return data

    @task
    def get_models():
        return [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression",
        ]

    @task(map_index_template="{{ custom_map_index }}")
    def train_model(
        data, model_name="RandomForestClassifier", test_size=0.2, random_state=42, **context
    ):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import os
        import joblib
        import numpy as np
        import pandas as pd

        model_mapping = {
            "RandomForestClassifier": RandomForestClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "LogisticRegression": LogisticRegression,
        }

        X = data.drop(columns=["churn", "signup_date", "churn_date", "last_updated", "uuid"])
        y = data["churn"]
        X = pd.get_dummies(X, drop_first=True)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = model_mapping[model_name]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ts = context["ts"]

        model_path = f"include/models/challengers/{ts}_{model_name}.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        acc = accuracy_score(y_test, y_pred)

        from airflow.operators.python import get_current_context

        context = get_current_context()
        context["custom_map_index"] = "Model: " + model_name + " Accuracy: " + str(acc)

        return {"accuracy": acc, "model_name": model_name, "ts": ts}

    @task.branch
    def check_if_champion_exists():
        import os

        if os.path.exists("include/models/champion"):

            if len(os.listdir("include/models/champion")) > 0:
                return "champion_exists"
            else:
                return "pick_champion_from_challengers"
        else:
            return "pick_champion_from_challengers"

    @task
    def pick_champion_from_challengers(models, **context):
        import os
        import shutil

        ts = context["ts"]

        best_model_data = max(models, key=lambda x: x["accuracy"])
        model_name = best_model_data["model_name"]

        src_path = f"include/models/challengers/{ts}_{model_name}.joblib"
        dest_path = f"include/models/champion/{ts}_{model_name}.joblib"

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(src_path, dest_path)

        return {
            "champion_model": model_name,
            "accuracy": best_model_data["accuracy"],
            "model_path": dest_path,
        }

    champion_exists = EmptyOperator(task_id="champion_exists")

    new_data = extract_latest_data()
    historical_data = get_historical_data()
    combined_data = combine_data(new_data, historical_data)
    new_data_loaded = load_new_data_to_db(new_data)
    models = get_models()
    training_data = feature_engineering(data=combined_data)
    trained_models = train_model.partial(data=training_data).expand(model_name=models)

    best_challenger = pick_champion_from_challengers(models=trained_models)

    chain(check_if_champion_exists(), [champion_exists, best_challenger])

    chain(combined_data, new_data_loaded)

    chain(trained_models, best_challenger)


model_refresh()
