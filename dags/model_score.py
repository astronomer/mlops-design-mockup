from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from pendulum import datetime


@dag(
    start_date=datetime(2024, 8, 1),
    schedule="@hourly",
    catchup=False,
)
def model_score():

    @task
    def get_data(**context):
        from include.utils import generate_inference_data

        today = context["logical_date"]

        new_data = generate_inference_data(num_customers=10, base_date=today)

        return new_data

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
    def get_champion_model():
        import os

        champion_dir = "include/models/champion"
        model_files = os.listdir(champion_dir)

        if not model_files:
            raise FileNotFoundError("No model found in the champion directory.")

        model_file = model_files[0]
        model_name = model_file.replace(".joblib", "")

        return model_name

    @task
    def get_challenger_models():
        import os
        import joblib

        challenger_dir = "include/models/challengers"
        model_files = os.listdir(challenger_dir)

        if not model_files:
            raise FileNotFoundError("No models found in the challenger directory.")

        model_files = [model.replace(".joblib", "") for model in model_files]
        return model_files

    @task
    def run_inference_champion(data, model_name, **context):
        import os
        import joblib
        import numpy as np
        import pandas as pd

        ts = context["ts"]

        model_path = f"include/models/champion/{model_name}.joblib"
        model = joblib.load(model_path)

        X = data.drop(
            columns=["churn", "signup_date", "churn_date", "last_updated", "uuid"]
        )
        X = pd.get_dummies(X, drop_first=True)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)

        y_pred = model.predict(X)

        predictions = pd.DataFrame({"prediction": y_pred})

        data["prediction"] = y_pred

        os.makedirs(f"include/predictions/{model_name}", exist_ok=True)
        data.to_csv(f"include/predictions/{model_name}/{ts}.csv", index=False)

        return {"model": model_name, "predictions": predictions}

    @task
    def run_inference_challenger(data, model_name, **context):
        import joblib
        import numpy as np
        import pandas as pd
        import os

        ts = context["ts"]

        model_path = f"include/models/challengers/{model_name}.joblib"
        model = joblib.load(model_path)

        X = data.drop(
            columns=["churn", "signup_date", "churn_date", "last_updated", "uuid"]
        )
        X = pd.get_dummies(X, drop_first=True)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True)

        y_pred = model.predict(X)

        predictions = pd.DataFrame({"prediction": y_pred})

        data["prediction"] = y_pred
        os.makedirs(f"include/predictions/{model_name}", exist_ok=True)
        data.to_csv(f"include/predictions/{model_name}/{ts}.csv", index=False)

        return {"model": model_name, "predictions": predictions}

    new_data = get_data()
    processed_data = feature_engineering(new_data)

    champion_model = get_champion_model()
    challenger_models = get_challenger_models()

    champion_inference = run_inference_champion(
        data=processed_data, model_name=champion_model
    )
    challenger_inferences = run_inference_challenger.partial(
        data=processed_data
    ).expand(model_name=challenger_models)


model_score()
