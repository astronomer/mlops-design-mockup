from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from pendulum import datetime
from airflow.operators.empty import EmptyOperator


@dag(
    start_date=datetime(2024, 8, 1),
    schedule="@daily",
    catchup=False,
)
def model_monitor():

    @task
    def mock_data_update():
        """This functions mocks that some customers have churned."""
        import os
        import pandas as pd
        import numpy as np

        np.random.seed(42)  # for consistency in results
        prediction_folder = "include/predictions"

        for filename in os.listdir(prediction_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(prediction_folder, filename)
                data = pd.read_csv(file_path)

                churn_probability = (
                    (data["monthly_charges"] > 80) * 0.3
                    + (data["tier_Individual"] == 1) * 0.5
                    + (data["support_calls"] > 5) * 0.4
                    + (data["product"] == "Product A") * 0.3
                ) / 2.5

                data["churn"] = (np.random.rand(len(data)) < churn_probability).astype(
                    int
                )

                data.to_csv(file_path, index=False)

    @task.branch
    def is_champion():
        # check if there is a champion model
        import os

        champion_dir = "include/models/champion"
        model_files = os.listdir(champion_dir)
        if len(model_files) > 0:
            return "get_champion_model"
        else:
            return "no_champion_exists"

    no_champion_exists = EmptyOperator(task_id="no_champion_exists")

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

    _is_champion = is_champion()
    _get_champion_model = get_champion_model()

    chain(_is_champion, [_get_champion_model, no_champion_exists])

    @task.branch
    def is_challenger():
        # check if there are challenger models
        import os

        challenger_dir = "include/models/challengers"
        model_files = os.listdir(challenger_dir)
        if len(model_files) > 0:
            return "get_challenger_models"
        else:
            return "no_challenger_exists"

    no_challenger_exists = EmptyOperator(task_id="no_challenger_exists")

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

    _is_challenger = is_challenger()
    _get_challenger_models = get_challenger_models()

    chain(_is_challenger, [_get_challenger_models, no_challenger_exists])

    @task
    def get_champion_inference_accuracy(model_name):
        import os
        import pandas as pd

        model_path = os.path.join("include/predictions", model_name)
        correct_predictions = 0
        total_predictions = 0

        for filename in os.listdir(model_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(model_path, filename)
                data = pd.read_csv(file_path)

                if "churn" in data.columns and "prediction" in data.columns:
                    correct_predictions += (data["churn"] == data["prediction"]).sum()
                    total_predictions += len(data)

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        return {"accuracy": accuracy, "model_name": model_name, "status": "champion"}

    @task(map_index_template="{{ custom_map_index }}")
    def get_challenger_inference_accuracy(model_name):
        import os
        import pandas as pd

        model_path = os.path.join("include/predictions", model_name)
        correct_predictions = 0
        total_predictions = 0

        for filename in os.listdir(model_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(model_path, filename)
                data = pd.read_csv(file_path)

                if "churn" in data.columns and "prediction" in data.columns:
                    correct_predictions += (data["churn"] == data["prediction"]).sum()
                    total_predictions += len(data)

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        from airflow.operators.python import get_current_context

        context = get_current_context()
        context["custom_map_index"] = (
            "Model: " + model_name + " Accuracy: " + str(accuracy)
        )

        return {"accuracy": accuracy, "model_name": model_name, "status": "challenger"}

    _mock_data_update = mock_data_update()
    _get_champion_inference_accuracy = get_champion_inference_accuracy(
        model_name=_get_champion_model
    )
    _get_challenger_inference_accuracy = get_challenger_inference_accuracy.expand(
        model_name=_get_challenger_models
    )

    chain(_mock_data_update, [_get_challenger_models, _get_champion_model])

    @task.branch
    def champion_vs_challengers(champion, challengers):
        champion_accuracy = champion["accuracy"]
        challenger_accuracies = [challenger["accuracy"] for challenger in challengers]

        if champion_accuracy > max(challenger_accuracies):
            return "keep_champion"
        else:
            # pick the best challenger as the new champion
            best_challenger = challengers[
                challenger_accuracies.index(max(challenger_accuracies))
            ]
            return "pick_new_champion"

    keep_champion = EmptyOperator(task_id="keep_champion", trigger_rule="none_failed")

    @task(trigger_rule="none_failed")
    def pick_new_champion(champion, challengers):
        import os
        import shutil

        best_challenger = max(challengers, key=lambda x: x["accuracy"])
        model_name = best_challenger["model_name"]

        src_path = f"include/models/challengers/{model_name}.joblib"
        dest_path = f"include/models/champion/{model_name}.joblib"

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(src_path, dest_path)

        # move old champion to challengers
        src_path = f"include/models/champion/{champion['model_name']}.joblib"
        dest_path = f"include/models/challengers/{champion['model_name']}.joblib"

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(src_path, dest_path)

        return {
            "champion_model": model_name,
            "accuracy": best_challenger["accuracy"],
            "model_path": dest_path,
        }

    _champion_vs_challengers = champion_vs_challengers(
        champion=_get_champion_inference_accuracy,
        challengers=_get_challenger_inference_accuracy,
    )

    _pick_new_champion = pick_new_champion(
        champion=_get_champion_inference_accuracy,
        challengers=_get_challenger_inference_accuracy,
    )

    chain(
        _champion_vs_challengers,
        [keep_champion, _pick_new_champion],
    )

    chain(no_challenger_exists, keep_champion)
    chain(no_champion_exists, _pick_new_champion)


model_monitor()
