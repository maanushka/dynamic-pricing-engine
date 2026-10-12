"""
The Price Is Not Random - Dynamic Pricing Demo
Tech Talk Tuesday | Team demo

Install once:  pip install numpy pandas scikit-learn matplotlib
Run:           python dynamic_pricing_demo.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

BASE_FARE = 30  # Rs, fixed booking fee
PER_KM = 12  # Rs per km


# ---------------------------------------------------------------
# STEP 1: Make fake ride data (the "real world" the AI learns from)
# ---------------------------------------------------------------
def true_surge(hour, rain, weekend, event):
    """Hidden rule of the world. The AI must DISCOVER this from data."""
    surge = 1.0
    if hour in (8, 9, 18, 19, 20):  # office rush hours
        surge += 0.5
    if hour >= 22 or hour <= 4:  # late night
        surge += 0.3
    if rain:
        surge += 0.6
    if weekend and 19 <= hour <= 23:  # weekend nights
        surge += 0.3
    if event:  # exam day / concert / match
        surge += 0.4
    return surge


def make_data(n=20000):
    rows = []
    for _ in range(n):
        km = np.random.uniform(1, 25)
        hour = np.random.randint(0, 24)
        rain = np.random.rand() < 0.2
        weekend = np.random.rand() < 0.3
        event = np.random.rand() < 0.1
        price = (BASE_FARE + PER_KM * km) * true_surge(hour, rain, weekend, event)
        price += np.random.normal(0, 8)  # real-world noise
        rows.append([km, hour, int(rain), int(weekend), int(event), price])
    return pd.DataFrame(
        rows, columns=["km", "hour", "rain", "weekend", "event", "price"]
    )


# ---------------------------------------------------------------
# STEP 2: Train the AI model
# ---------------------------------------------------------------
def train_model():
    df = make_data()
    X = df[["km", "hour", "rain", "weekend", "event"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    model = RandomForestRegressor(n_estimators=150, random_state=1)
    model.fit(X_train, y_train)
    score = r2_score(y_test, model.predict(X_test))
    return model, score, X.columns.tolist()


# ---------------------------------------------------------------
# STEP 3: Live demo functions
# ---------------------------------------------------------------
def predict_price(model, km, hour, rain, weekend, event):
    row = pd.DataFrame(
        [[km, hour, rain, weekend, event]],
        columns=["km", "hour", "rain", "weekend", "event"],
    )
    return float(model.predict(row)[0])


def ask_yes_no(question):
    return 1 if input(question + " (y/n): ").strip().lower() == "y" else 0


def live_demo(model):
    print("\n--- Book a cab ---")
    km = float(input("Distance in km (e.g. 8): "))
    hour = int(input("Hour of day, 0-23 (e.g. 19 for 7 PM): "))
    rain = ask_yes_no("Is it raining?")
    weekend = ask_yes_no("Is it a weekend?")
    event = ask_yes_no("Exam day / concert / match nearby?")

    normal = predict_price(model, km, 14, 0, 0, 0)  # calm 2 PM, dry, weekday
    price = predict_price(model, km, hour, rain, weekend, event)
    multiplier = price / normal

    print("\n==============================")
    print(f"Normal price (calm day): Rs {normal:.0f}")
    print(f"YOUR price right now   : Rs {price:.0f}")
    print(f"Surge multiplier       : {multiplier:.2f}x")
    print("==============================")
    if multiplier > 1.4:
        print("High surge! The AI sees high demand and raises the price.")
    elif multiplier > 1.1:
        print("Mild surge.")
    else:
        print("No surge. Good time to travel!")


def show_chart(model):
    import matplotlib.pyplot as plt

    hours = list(range(24))
    dry = [predict_price(model, 8, h, 0, 0, 0) for h in hours]
    wet = [predict_price(model, 8, h, 1, 0, 0) for h in hours]

    plt.figure(figsize=(10, 5))
    plt.plot(hours, dry, marker="o", label="Dry day")
    plt.plot(hours, wet, marker="o", label="Rainy day")
    plt.title("Price of an 8 km ride through the day (AI prediction)")
    plt.xlabel("Hour of day")
    plt.ylabel("Price (Rs)")
    plt.xticks(hours)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("price_chart.png", dpi=150)
    print("Chart saved as price_chart.png")
    plt.show()


def what_matters(model, features):
    print("\n--- What did the AI learn matters most? ---")
    pairs = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    for name, imp in pairs:
        print(f"{name:8s} {'#' * int(imp * 50)} {imp * 100:.0f}%")


# ---------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------
def main():
    print("Training the AI on 20,000 fake rides...")
    model, score, features = train_model()
    print(f"Done! Model accuracy (R2 score): {score:.2f}")

    while True:
        print("\n1) Book a cab (live demo)")
        print("2) Show price-vs-hour chart")
        print("3) What does the AI think matters?")
        print("4) Quit")
        choice = input("Choose: ").strip()
        if choice == "1":
            live_demo(model)
        elif choice == "2":
            show_chart(model)
        elif choice == "3":
            what_matters(model, features)
        elif choice == "4":
            break


if __name__ == "__main__":
    main()
