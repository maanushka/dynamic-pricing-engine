"""
Dynamic Pricing Engine & Ethical Fare System
Author: Manushka
Project: Algorithmic Pricing with Fair Surge Caps
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. GENERATE DATASET
# ==========================================
np.random.seed(42)
TOTAL_RIDES = 20000

rain_mm = np.random.exponential(scale=2.5, size=TOTAL_RIDES)
traffic_index = np.random.uniform(0.1, 1.0, size=TOTAL_RIDES)
hour_of_day = np.random.randint(0, 24, size=TOTAL_RIDES)
ride_requests = np.random.poisson(lam=90, size=TOTAL_RIDES)
available_cabs = np.clip(
    np.random.poisson(lam=50, size=TOTAL_RIDES), a_min=5, a_max=None
)

# Market surge formula
demand_supply_ratio = ride_requests / available_cabs
peak_hours = ((hour_of_day >= 8) & (hour_of_day <= 10)) | (
    (hour_of_day >= 17) & (hour_of_day <= 20)
)

calculated_surge = (
    1.0
    + 0.40 * np.maximum(0, demand_supply_ratio - 1.0)
    + 0.30 * traffic_index
    + 0.15 * (rain_mm > 5.0)
    + 0.15 * peak_hours
)
surge_multiplier = np.clip(calculated_surge, a_min=1.0, a_max=3.5)

data = pd.DataFrame(
    {
        "rain_mm": rain_mm,
        "traffic_index": traffic_index,
        "hour_of_day": hour_of_day,
        "ride_requests": ride_requests,
        "available_cabs": available_cabs,
        "surge_multiplier": surge_multiplier,
    }
)

print("Dataset created successfully!")
print(data.head())

# ==========================================
# 2. TRAIN THE MACHINE LEARNING MODEL
# ==========================================
X = data[["rain_mm", "traffic_index", "hour_of_day", "ride_requests", "available_cabs"]]
y = data["surge_multiplier"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining the Gradient Boosting model...")
model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Training Complete!")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ==========================================
# 3. PRICING & ETHICAL CAP LOGIC
# ==========================================
BASE_FARE = 125  # Base price in INR (₹)
MAX_SURGE_CAP = 1.5  # Maximum fair surge multiplier (1.5x)


def calculate_ride_fare(rain, traffic, hour, requests, cabs):
    input_data = pd.DataFrame(
        [[rain, traffic, hour, requests, cabs]],
        columns=[
            "rain_mm",
            "traffic_index",
            "hour_of_day",
            "ride_requests",
            "available_cabs",
        ],
    )

    predicted_surge = float(model.predict(input_data)[0])
    predicted_surge = max(1.0, round(predicted_surge, 2))

    uncapped_fare = round(BASE_FARE * predicted_surge)
    uncapped_driver_earning = round(uncapped_fare * 0.75)

    is_capped = predicted_surge > MAX_SURGE_CAP
    fair_surge = MAX_SURGE_CAP if is_capped else predicted_surge
    fair_fare = round(BASE_FARE * fair_surge)

    commission_rate = 0.05 if is_capped else 0.25
    fair_driver_earning = round(fair_fare * (1 - commission_rate))

    user_savings = uncapped_fare - fair_fare

    return {
        "uncapped_surge": predicted_surge,
        "uncapped_fare": uncapped_fare,
        "uncapped_driver": uncapped_driver_earning,
        "fair_surge": fair_surge,
        "fair_fare": fair_fare,
        "fair_driver": fair_driver_earning,
        "savings": user_savings,
        "capped_triggered": is_capped,
    }


def display_fare_summary(title, rain, traffic, hour, requests, cabs):
    result = calculate_ride_fare(rain, traffic, hour, requests, cabs)

    print("\n" + "=" * 50)
    print(f"Ride Scenario: {title}")
    print(
        f"Inputs -> Rain: {rain}mm | Congestion: {traffic:.2f} | Dem/Supp: {requests}/{cabs}"
    )
    print("-" * 50)
    print(
        f"Original Market Surge : {result['uncapped_surge']}x  --> Total Fare: ₹{result['uncapped_fare']}"
    )
    print(f"Original Driver Payout: ₹{result['uncapped_driver']} (Platform takes 25%)")
    print("-" * 50)
    print(
        f"Regulated Fair Surge  : {result['fair_surge']}x  --> Total Fare: ₹{result['fair_fare']}"
    )
    print(f"Regulated Driver Cut  : ₹{result['fair_driver']} (Platform subsidizes fee)")
    if result["capped_triggered"]:
        print(
            f"Ethical Guardrail     : Capped at {MAX_SURGE_CAP}x (Passenger saved ₹{result['savings']}!)"
        )
    else:
        print("Ethical Guardrail     : Normal market pricing (No cap needed)")
    print("=" * 50)


# ==========================================
# 4. RUN TEST SCENARIOS & INTERACTIVE PROMPT
# ==========================================
if __name__ == "__main__":
    display_fare_summary(
        "Normal Afternoon", rain=0.0, traffic=0.2, hour=14, requests=45, cabs=50
    )
    display_fare_summary(
        "Monsoon Evening Rush", rain=15.0, traffic=0.85, hour=19, requests=180, cabs=40
    )

    print("\n--- Try It Yourself ---")
    try:
        r = float(input("Enter rainfall in mm (e.g. 10): "))
        t = float(input("Enter traffic index between 0.1 and 1.0 (e.g. 0.8): "))
        h = int(input("Enter current hour (0-23): "))
        req = int(input("Enter active ride requests (e.g. 150): "))
        cab = int(input("Enter available cabs (e.g. 35): "))

        display_fare_summary(
            "Custom Live Test", rain=r, traffic=t, hour=h, requests=req, cabs=cab
        )
    except ValueError:
        print("Invalid input entered. Demo finished.")
