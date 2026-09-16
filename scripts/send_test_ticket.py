"""Send a single random ticket to the local /predict API for a manual smoke test."""

from __future__ import annotations

import json
import random
import sys
import urllib.request

API_URL = "http://localhost:8000/predict"

SAMPLE_TICKETS = [
    "hi team, i was charged twice for my subscription this month",
    "the app crashes every time i try to upload a profile picture",
    "can you help me reset my password, i never got the reset email",
    "your product is amazing, just wanted to say thanks for the great support",
    "i'd like to cancel my account and get a refund for this month",
    "the dashboard is loading really slowly since yesterday's update",
    "how do i export my data to csv? i can't find the option anywhere",
    "my shipment says delivered but i never received the package",
]


def main() -> None:
    text = random.choice(SAMPLE_TICKETS)
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    print(f"POST {API_URL}\n  text: {text!r}")
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
            print(f"  status: {resp.status}")
            print(f"  response: {json.dumps(body, indent=2)}")
    except urllib.error.HTTPError as exc:
        print(f"  status: {exc.code}")
        print(f"  response: {exc.read().decode()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
