import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid

def generate_churn_data(
    num_customers=1000, base_date=datetime(2024, 1, 1), filename="churn_data.csv"
):
    np.random.seed(42)

    uuids = [str(uuid.uuid4()) for _ in range(num_customers)]

    monthly_charges = np.random.uniform(20, 120, num_customers)
    support_calls = np.random.randint(0, 10, num_customers)
    customer_tier = np.random.choice(
        ["Individual", "Team", "Enterprise"], num_customers
    )
    product = np.random.choice(["Product A", "Product B"], num_customers)
    operating_system = np.random.choice(["Windows", "Linux", "Mac"], num_customers)

    signup_dates = [
        base_date - timedelta(days=np.random.randint(0, 365 * 3))
        for _ in range(num_customers)
    ]

    churn_probability = (
        (monthly_charges > 80) * 0.3
        + (customer_tier == "Individual") * 0.5
        + (support_calls > 5) * 0.4
        + (product == "Product A") * 0.3
    ) / 2.5

    churn = (np.random.rand(num_customers) < churn_probability).astype(int)

    churn_dates = []
    tenures = []

    for i in range(num_customers):
        if churn[i] == 1:

            churn_date = signup_dates[i] + timedelta(days=np.random.randint(30, 365))
            churn_dates.append(churn_date)
            tenure = (churn_date - signup_dates[i]).days // 30
            tenures.append(tenure)
        else:

            churn_dates.append(None)
            tenure = (base_date - signup_dates[i]).days // 30
            tenures.append(tenure)

    base_date = base_date.strftime("%Y-%m-%d")
    churn_dates = [
        date.strftime("%Y-%m-%d") if date is not None else None for date in churn_dates
    ]
    signup_dates = [date.strftime("%Y-%m-%d") for date in signup_dates]

    data = pd.DataFrame(
        {
            "uuid": uuids,
            "signup_date": signup_dates,
            "tenure": tenures,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "tier": customer_tier,
            "product": product,
            "operating_system": operating_system,
            "churn": churn,
            "churn_date": churn_dates,
            "last_updated": base_date,
        }
    )

    data.to_csv(filename, index=False)


generate_churn_data(base_date=datetime(2024, 10, 28))
