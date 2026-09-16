# Ticket routing — a text classifier in production

A deliberately small support-ticket classifier (billing / technical / account / shipping),
deployed the way a production model should be: trained by a reproducible script, served
behind a validated API, containerised, watched by Prometheus and Grafana, and gated in CI
so a bad model cannot ship.

The model is boring on purpose. Everything around the model is the point — and with text,
the monitoring is where it gets genuinely interesting.

```
        train.py ──► artifacts/ ──► quality gate ──► Docker image ──► FastAPI ──► /metrics
       (offline)     model.joblib    (CI/pytest)      (immutable)      (online)       │
                     metrics.json         │            same artifact                  ▼
                     reference_sample.csv │            the gate passed           Prometheus ──► Grafana
                     drift_reference.json │                                            │
                                     fails ──► nothing ships                      alerts.yml
```

## Quick start

```bash
make install    # deps + editable install
make train      # writes artifacts/
make test       # 51 tests, including the model quality gate
make up         # API + Prometheus + Grafana via docker compose
```

| What | Where |
|---|---|
| API docs | http://localhost:8000/docs |
| Raw metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

Then watch monitoring actually work:

```bash
make traffic    # normal tickets   → all five drift signals stay at zero
make drift      # shifted tickets  → four of five light up, alerts go pending
```

Measured on this machine:

| | normal traffic | shifted traffic |
|---|---|---|
| signals firing | 0 / 5 | 4 / 5 |
| embedding distance (threshold 0.052) | 0.045 | 0.192 |
| out-of-vocabulary rate | 0.0% | 11.7% |
| class-mix distance | 0.05 | 0.30 |
| routing mix | roughly even | billing 55% |

## The five pieces, and why each exists

### 1. Training is a script, not a notebook

`tickets/train.py` is the only way a model gets made. Deterministic (one seed), it writes
its own metrics next to the model and stamps every model with a version
(`20260915-202101-a1b2c3d` — UTC timestamp + git SHA). A model you cannot trace back to a
commit is a model you cannot debug six months from now.

It emits four artifacts, and the rest of the system depends on all four:

| Artifact | Consumed by | Why |
|---|---|---|
| `model.joblib` | the API | pipeline + the LSA embedder used for drift |
| `metrics.json` | the CI quality gate | held-out scores, per-class F1, confusion matrix |
| `reference_sample.csv` | drift monitoring | length baseline, and what training text looked like |
| `drift_reference.json` | drift monitoring | embedding centroid, class mix, **calibrated thresholds** |

**Normalisation and vectorisation live inside the sklearn `Pipeline`** — `normalize →
TfidfVectorizer → LogisticRegression`. This is the single most common way real NLP systems
break: text cleaned one way in the training notebook and another way in the serving code,
so the model sees inputs it was never fitted on. If it is in the pipeline, it cannot diverge.

### 2. Serving validates at the edge — and text needs more of it than numbers do

`tickets/api.py`. The service-shape basics first:

- **The model loads once, at startup** (FastAPI `lifespan`), never inside a request.
  Cold-loading a vectorizer per request is the classic way to destroy p99 latency.
- **Startup fails loudly** if the artifact is missing. A service running without its model
  is worse than one that refuses to start.
- **`/health` vs `/ready`**: liveness says the process is alive; readiness says it can
  actually serve. Orchestrators route traffic on readiness; conflating them sends traffic
  to a pod that isn't ready.

Then the text-specific parts, which is where I made the judgement calls you left to me:

- **A hard length cap** (`MAX_TEXT_CHARS = 5000`). An unbounded string field is a cost and
  memory problem — with a transformer it is also a latency cliff.
- **Blank input is rejected.** A model asked to classify `"   "` will answer confidently
  and meaninglessly. `min_length` alone does not catch whitespace.
- **PII is scrubbed before anything else touches the text** (`tickets/text.py`): emails,
  URLs, order references and phone-shaped digit runs become placeholders. Two reasons, and
  the second is the one people forget. First, none of it should reach logs or the drift
  reference sample sitting on disk. Second, it is pure noise to the model — a phone number
  is never evidence that a ticket is about billing, it just inflates the vocabulary.
- **Unicode normalisation and case folding**, so `"MY PARCEL"` and `"my parcel"` are one
  thing. `test_invariance_to_formatting` pins this through the model.

Because normalisation is inside the pipeline, all of the above applies identically at
training time. That is the property the tests in `tests/test_text.py` exist to protect.

### 3. Monitoring text drift: five signals, none of which need labels

Tabular drift is easy — compare each column's distribution. **Text has no columns.** This
is the part of the project worth your attention.

The service watches five different windows onto the same question:

| Signal | Statistic | Catches |
|---|---|---|
| **Embedding drift** | cosine distance between the live window's centroid and the training centroid, in LSA space | meaning moving, in general |
| **OOV rate** | share of tokens the vectorizer has never seen | a product rename, new slang, another language |
| **Length drift** | KS test on token counts | a new UI, a bot, a changed template |
| **Class-mix drift** | total variation distance between predicted mix and training mix | an incident flooding one queue |
| **Confidence drift** | mean confidence, low-confidence rate | the model losing its footing |

None of them needs ground truth, which is the point: labels arrive weeks late, if ever.
They also disagree usefully — on the shifted traffic above, OOV and class-mix fire loudly
while confidence does not move at all, because that particular shift changes *vocabulary
and volume*, not ambiguity. The model stays confident while routing text it has never
seen, which is exactly the failure mode people assume confidence will catch. It doesn't.
That's why there are five.

**Thresholds are calibrated at training time, not guessed.** `train.py` bootstraps 500
windows out of the training set, measures how far each window's centroid lands from the
overall centroid by pure chance, and takes the 99.5th percentile (here: 0.0519). The OOV
bar is derived the same way — three times the training baseline, with a floor. Both are
written to `drift_reference.json` and re-exported as Prometheus gauges, so the alert rule
is `distance > threshold` rather than a magic number in YAML, and **a retrain
automatically re-derives the bar for the model that is actually serving.**

The embedder is `TruncatedSVD` over the *same* TF-IDF the classifier uses. No second model,
no download, no extra dependency — and drift is measured in the model's own view of the text.

> One bug worth keeping: my first version measured the mean distance of *each document* to
> the training centroid. That is a measure of spread, not location. Drifted traffic that
> was more uniform than training data scored **lower** — 0.741 → 0.698 — so real drift read
> as an improvement. Centroid-to-centroid fixed it (0.045 → 0.192), and
> `test_embedding_drift_rises_under_shift` fails if anyone changes it back. A monitor that
> is quietly wrong is worse than no monitor, because you trust it.

`monitoring/alerts.yml` turns metrics into alerts. Drift alerts require **two signals
firing for ten minutes** — one signal on a quiet afternoon is noise, two at once is a
pattern.

### 4. CI is a quality gate, not just a test run

`.github/workflows/ci.yml`: lint → train → **quality gate** → contract tests → build.
CI answers "is this commit sound?" and **never pushes or deploys**; releasing is
[§6](#6-cd-is-separate-and-tag-triggered).

`tests/test_model_quality.py` asserts the freshly trained model clears `MIN_MACRO_F1`,
`MIN_ACCURACY` and — separately — `MIN_PER_CLASS_F1` **for every class**. Macro-F1 and
per-class floors, not accuracy: accuracy lets a model quietly abandon the smallest queue
and still look fine. It also checks behaviour, not just aggregates — one textbook ticket
per class must route correctly, and formatting must not change a decision.

The image build job `needs: test`. **A model that fails the gate never becomes an image.**
That one line of YAML is the difference between "we have tests" and "we have a gate".

The gate only means something if the model it validated is the model that ships. CI trains
**once**, uploads `artifacts/`, and the build job downloads that exact artifact and copies
it into the image — so the gate and the image are talking about the same model. The smoke
test then asserts the running container reports the `model_version` the gate signed off on.
(Training inside `docker build` instead would quietly produce a *second* model, leaving the
gate to validate an artifact that never reaches production.)

The thresholds sit at 0.80 against a model that scores **0.845**, because the ceiling on
this corpus is around 0.86 by construction: 4% of labels are noise, 12% of tickets are
genuinely ambiguous between two queues, 8% are vague one-liners. An unreachable gate gets
disabled, and a disabled gate protects nothing. Thresholds live in version control, so
moving the bar is a reviewable commit rather than someone's Friday-afternoon judgement.

### 5. The image is the unit of deployment

The model is baked into the image — copied in from the training run that passed the quality
gate, not retrained during the build. The image tag *is* the model version: reproducible,
immutable, rollback = redeploy the previous tag.

Because the build copies rather than trains, `artifacts/` must exist first. `make image`
and `make up` train for you; a bare `docker build` on a clean checkout fails with a clear
message instead of shipping an image with no model in it.

### 6. CD is separate, and tag-triggered

`.github/workflows/cd.yml` runs on a `v*` tag, not on every merge — because "this code is
good" and "ship it now" are different decisions, and only one of them should be made by
merging a PR.

```bash
git tag v1.0.0 && git push origin v1.0.0     # release
```

train → **quality gate** → build → smoke test → push to `ghcr.io`. Three properties worth
naming:

- **It retrains rather than reusing CI's artifact.** CI's run may be days old and its
  artifact expires, so a release that depended on it would not be reproducible from its
  own tag. CD's release is reproducible from the tag alone — and the gate runs again, so
  nothing ships unvalidated.
- **The smoke test runs before the push.** The image is built with `load: true`, tested
  locally, and only then pushed. A broken release never reaches the registry.
- **One tag names one model.** The git tag, the image tag and the `model_version` all
  refer to the same artifact, recorded on the image as `ml.model.version` /
  `ml.model.macro_f1` labels. Rollback is `docker pull` of the previous tag — no rebuild,
  no retrain.

`scripts/smoke_test.sh` is shared by both workflows so they cannot drift: it asserts the
running container reports the *exact* `model_version` the gate signed off on.

CD stops at a pushed image. There is no deploy step because this repo has nowhere to
deploy to, and a placeholder that pretends otherwise would be the dishonest kind of
scaffolding. Adding one means a `deploy:` job with `environment: production` (for the
approval gate and deployment history) that pulls the tag.

The trade-off, stated plainly: retraining requires a rebuild, and the same image cannot be
repointed at a newer model. The alternative is a **model registry** (MLflow, S3 + metadata
store) that the service pulls from at startup, so models ship independently of code. That
is the right answer once retraining is frequent, and the wrong kind of complexity while it
isn't. This repo takes the simple option and tells you where it stops.

## Layout

```
tickets/
  config.py       every threshold and path in one place
  data.py         synthetic ticket corpus (with a `shift` knob to simulate drift)
  text.py         normalisation + PII scrubbing, mounted inside the pipeline
  train.py        the only way a model gets made; calibrates drift thresholds
  model.py        artifact loading, prediction, embedding
  schemas.py      request/response contracts
  monitoring.py   Prometheus metrics + the five-signal drift monitor
  api.py          FastAPI service
tests/            text, data, quality gate, API contracts, drift detection (51 tests)
monitoring/       prometheus.yml, alerts.yml, Grafana provisioning + dashboard
scripts/          traffic simulator (--shift to drift it), shared image smoke test
.github/workflows/
  ci.yml          every PR + main: lint, types, gate, build. Never pushes.
  cd.yml          `v*` tags: retrain, re-gate, build, smoke test, push to GHCR
```

The corpus is synthetic and generated locally: deterministic, no network, and `shift`
moves vocabulary, length and class mix at once — the three ways text distributions
actually move in production. Swap `data.py` for a real loader and nothing else changes;
that is the interface working.

## What this deliberately does not have

Know the gaps; they are the rest of the MLOps curriculum:

- **Ground-truth / performance monitoring.** Drift is a proxy. Real accuracy needs a
  feedback endpoint capturing which queue a ticket was *actually* handled by, joined back
  to the prediction, tracking live macro-F1. Hardest part of ML in production, and the
  most valuable.
- **A model registry + experiment tracking** (MLflow). See §5.
- **Automated retraining.** Today: drift alert → human investigates → rerun training →
  gate decides. Automating the trigger before you trust the gate just ships bad models faster.
- **A stronger model.** TF-IDF + logistic regression is the right starting point; a
  fine-tuned transformer would beat it and change every operational constraint (image
  size, GPU, latency, cost). Start here, and let the monitoring tell you when it isn't enough.
- **Kubernetes**, autoscaling, canary/shadow deploys. Compose teaches the shape; k8s
  teaches the operations.
- **Real auth, rate limiting, TLS.** The API is wide open.
- **Explainability and fairness slicing** — often a regulatory requirement, not a nicety.
