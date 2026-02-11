import random
import time
import requests

API_URL = "http://127.0.0.1:8000/predict"

RESIDENCE = ["Owned", "Rented", "Mortgage"]
PURPOSE = ["Education", "Home", "Auto", "Personal"]
LOAN_TYPE = ["Unsecured", "Secured"]

def random_payload():
    income = random.randint(300_000, 3_000_000)
    loan_amount = random.randint(100_000, 5_000_000)

    return {
        "age": random.randint(18, 70),
        "income": float(income),
        "loan_amount": float(loan_amount),
        "loan_tenure_months": random.choice([12, 24, 36, 48, 60]),
        "avg_dpd_per_delinquency": float(random.randint(0, 60)),
        "delinquency_ratio": float(random.randint(0, 100)),
        "credit_utilization_ratio": float(random.randint(0, 100)),
        "num_open_accounts": random.randint(1, 10),
        "residence_type": random.choice(RESIDENCE),
        "loan_purpose": random.choice(PURPOSE),
        "loan_type": random.choice(LOAN_TYPE),
    }

def main(n=100, sleep=0.05):
    ok = 0
    for i in range(n):
        payload = random_payload()
        r = requests.post(API_URL, json=payload, timeout=15)

        if r.status_code == 200:
            ok += 1
        else:
            print("Failed:", r.status_code, r.text)

        if (i + 1) % 10 == 0:
            print(f"Sent {i+1}/{n} (success={ok})")

        time.sleep(sleep)

    print(f"Done ✅ success={ok}/{n}")

if __name__ == "__main__":
    main(n=50, sleep=0.03)
