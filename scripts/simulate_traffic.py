"""Send traffic at the running service so the dashboards have something to show.

  python scripts/simulate_traffic.py                 # normal ticket traffic
  python scripts/simulate_traffic.py --shift 1.0     # new vocabulary, longer tickets,
                                                     # and a billing incident

Run the normal case first and watch the drift panels sit at zero, then run the shifted
case and watch them light up. That contrast is the whole point of the exercise.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter

from tickets import data


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--n", type=int, default=400, help="number of requests")
    parser.add_argument("--rps", type=float, default=25.0, help="requests per second")
    parser.add_argument(
        "--shift", type=float, default=0.0, help="0 = training distribution, 1 = fully drifted"
    )
    parser.add_argument(
        "--bad-rate", type=float, default=0.0, help="fraction of deliberately invalid requests"
    )
    args = parser.parse_args()

    tickets = data.generate(
        n_rows=args.n, seed=random.randint(0, 10_000), shift=args.shift
    )["text"].tolist()

    delay = 1.0 / args.rps if args.rps > 0 else 0.0
    routed: Counter[str] = Counter()
    rejected = failed = 0
    total_confidence = 0.0

    for text in tickets:
        payload = {"text": "" if random.random() < args.bad_rate else text}
        try:
            body = post(args.url, payload)
            routed[body["label"]] += 1
            total_confidence += body["confidence"]
        except urllib.error.HTTPError as error:
            rejected += 1 if error.code == 422 else 0
            failed += 0 if error.code == 422 else 1
        except Exception:
            failed += 1
        time.sleep(delay)

    served = sum(routed.values())
    mean_confidence = total_confidence / served if served else 0.0
    mix = ", ".join(f"{label} {count / served:.0%}" for label, count in sorted(routed.items()))
    print(
        f"shift={args.shift}  ok={served}  rejected(422)={rejected}  failed={failed}\n"
        f"  mean confidence={mean_confidence:.3f}\n  routing mix: {mix}"
    )


if __name__ == "__main__":
    main()
