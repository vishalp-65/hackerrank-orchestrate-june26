# Evaluation Report — Multi-Modal Evidence Review

Provider: **foundry**  |  Prompt version: **v4**  |  Configs compared: claude-opus-4-8, claude-sonnet-4-6

## 1. Dataset (sample / dev split)

- Rows: **20** labeled claims from `dataset/sample_claims.csv`
- claim_object: {'car': 8, 'laptop': 6, 'package': 6}
- claim_status (gold): {'supported': 13, 'contradicted': 5, 'not_enough_information': 2}
- risk-flag tokens (gold): {'none': 11, 'manual_review_required': 7, 'user_history_risk': 6, 'damage_not_visible': 4, 'claim_mismatch': 3, 'wrong_angle': 1, 'blurry_image': 1, 'non_original_image': 1, 'cropped_or_obstructed': 1, 'wrong_object': 1, 'text_instruction_present': 1}

## 2. Configuration comparison

| Metric | opus | sonnet |
|---|---|---|
| Composite score | 0.682 | 0.551 |
| claim_status accuracy | 0.800 | 0.600 |
| claim_status macro-F1 | 0.697 | 0.492 |
| risk_flags micro-F1 | 0.754 | 0.579 |
| risk_flags exact-match | 0.550 | 0.250 |
| evidence_standard_met acc | 0.900 | 0.750 |
| valid_image acc | 0.950 | 0.600 |
| issue_type macro-F1 | 0.395 | 0.447 |
| object_part macro-F1 | 0.854 | 0.792 |
| severity macro-F1 | 0.404 | 0.281 |
| supporting_image_ids exact | 0.800 | 0.650 |
| API calls (sample) | 20 | 20 |
| Input tokens (sample) | 40274 | 36385 |
| Output tokens (sample) | 15738 | 13919 |
| Cache-read tokens (sample) | 88111 | 58656 |
| Est. cost USD (sample) | $0.7361 | $0.3905 |
| Wall time s (sample) | 55.4 | 80.5 |

**Selected configuration: `claude-opus-4-8`** (highest composite = 0.682).

## 3. Per-field accuracy — opus

| Field | Accuracy |
|---|---|
| evidence_standard_met | 0.900 |
| valid_image | 0.950 |
| claim_status | 0.800 |
| issue_type | 0.500 |
| object_part | 0.850 |
| severity | 0.450 |
| risk_flags (micro-F1 / exact) | 0.754 / 0.550 |
| supporting_image_ids (F1 / exact) | 0.857 / 0.800 |

## 4. Error analysis — opus (17 rows with diffs)

- row 0 (user_001, car): severity: pred=`high` vs gold=`medium`
- row 1 (user_002, car): issue_type: pred=`dent` vs gold=`scratch`; severity: pred=`medium` vs gold=`low`
- row 2 (user_004, car): issue_type: pred=`glass_shatter` vs gold=`crack`; severity: pred=`high` vs gold=`medium`
- row 3 (user_007, car): issue_type: pred=`crack` vs gold=`broken_part`
- row 4 (user_005, car): issue_type: pred=`none` vs gold=`scratch`; severity: pred=`none` vs gold=`low`; risk_flags: pred=`claim_mismatch;cropped_or_obstructed;damage_not_visible;manual_review_required;user_history_risk;wrong_object_part` vs gold=`claim_mismatch;user_history_risk;manual_review_required`; supporting_image_ids: pred=`img_2` vs gold=`img_1`
- row 5 (user_006, car): object_part: pred=`body` vs gold=`headlight`; risk_flags: pred=`damage_not_visible;low_light_or_glare;wrong_angle` vs gold=`wrong_angle;damage_not_visible`
- row 7 (user_008, car): evidence_standard_met: pred=`false` vs gold=`true`; claim_status: pred=`not_enough_information` vs gold=`contradicted`; issue_type: pred=`unknown` vs gold=`broken_part`; object_part: pred=`body` vs gold=`front_bumper`; severity: pred=`unknown` vs gold=`high`; risk_flags: pred=`claim_mismatch;manual_review_required;non_original_image;user_history_risk;wrong_object_part` vs gold=`claim_mismatch;non_original_image;user_history_risk;manual_review_required`; supporting_image_ids: pred=`none` vs gold=`img_1`
- row 8 (user_009, laptop): issue_type: pred=`glass_shatter` vs gold=`crack`; severity: pred=`high` vs gold=`medium`
- row 9 (user_010, laptop): severity: pred=`high` vs gold=`medium`
- row 10 (user_011, laptop): issue_type: pred=`water_damage` vs gold=`stain`
- row 11 (user_012, laptop): risk_flags: pred=`blurry_image;low_light_or_glare` vs gold=`none`
- row 12 (user_018, laptop): issue_type: pred=`glass_shatter` vs gold=`crack`; severity: pred=`high` vs gold=`medium`
- row 13 (user_020, laptop): risk_flags: pred=`claim_mismatch;damage_not_visible;manual_review_required;user_history_risk` vs gold=`damage_not_visible;user_history_risk;manual_review_required`
- row 14 (user_015, package): claim_status: pred=`contradicted` vs gold=`supported`; issue_type: pred=`none` vs gold=`crushed_packaging`; severity: pred=`none` vs gold=`medium`; risk_flags: pred=`claim_mismatch` vs gold=`none`
- row 17 (user_032, package): risk_flags: pred=`damage_not_visible;manual_review_required;non_original_image` vs gold=`cropped_or_obstructed;damage_not_visible;manual_review_required`
- row 18 (user_033, package): evidence_standard_met: pred=`false` vs gold=`true`; claim_status: pred=`not_enough_information` vs gold=`contradicted`; object_part: pred=`box` vs gold=`unknown`; severity: pred=`unknown` vs gold=`low`; risk_flags: pred=`damage_not_visible;manual_review_required;user_history_risk;wrong_object` vs gold=`wrong_object;claim_mismatch;user_history_risk;manual_review_required`; supporting_image_ids: pred=`none` vs gold=`img_1`
- row 19 (user_034, package): valid_image: pred=`false` vs gold=`true`; claim_status: pred=`supported` vs gold=`contradicted`; issue_type: pred=`torn_packaging` vs gold=`none`; severity: pred=`medium` vs gold=`none`; risk_flags: pred=`manual_review_required;non_original_image;text_instruction_present;user_history_risk` vs gold=`damage_not_visible;text_instruction_present;user_history_risk;manual_review_required`; supporting_image_ids: pred=`img_1` vs gold=`img_1;img_2`

## 5. Operational analysis

Pricing assumptions (USD per 1M tokens): claude-opus-4-8 = $5.0/25.0, claude-sonnet-4-6 = $3.0/15.0, claude-haiku-4-5 = $1.0/5.0. Cache economics: fresh input ×1.0, cache write ×1.25, cache read ×0.1. Image tokens ≈ width×height/750; images are downscaled to a 1568px long edge before sending.

### Sample run — claude-opus-4-8
- Model calls: 20 (cache hits: 0); fallback rows: 0
- Tokens — input 40274, output 15738, cache-read 88111, cache-write 15549
- Images processed: 29 (missing 0); source formats: {'JPEG': 18, 'WEBP': 6, 'PNG': 5}
- Est. cost: $0.7361; wall 55.4s; per-call latency p50 10.2s / p95 12.9s

### Sample run — claude-sonnet-4-6
- Model calls: 20 (cache hits: 0); fallback rows: 0
- Tokens — input 36385, output 13919, cache-read 58656, cache-write 14664
- Images processed: 29 (missing 0); source formats: {'JPEG': 18, 'WEBP': 6, 'PNG': 5}
- Est. cost: $0.3905; wall 80.5s; per-call latency p50 14.8s / p95 18.1s

### Test run — actuals (dataset/claims.csv → output.csv)
- Model: claude-opus-4-8 (foundry); claims: 44; API calls: 44; cache hits: 0; fallbacks: 0
- Tokens — input 101334, output 37654, cache-read 207320, cache-write 20732
- Images processed: 82 (missing 0); formats: {'JPEG': 49, 'AVIF': 8, 'PNG': 14, 'WEBP': 11}
- Est. cost: $1.6813; wall 175.5s; latency p50 10.8s / p95 72.4s

### TPM/RPM, batching, throttling, caching, retries

- **Batching/throttling:** bounded `ThreadPoolExecutor(max_workers=4)` keeps concurrent requests well under provider RPM limits; the modest token volume (~thousands of input tokens/call) stays far below typical TPM ceilings.
- **Caching:** (1) a content-addressed **disk cache** of raw perception keyed by inputs+prompt_version+model — re-runs and policy edits cost **zero** API calls; (2) **prompt caching** (ephemeral `cache_control`) on the static system+requirements prefix, served from cache after the first call (see cache-read tokens above).
- **Retries:** exponential backoff with jitter on 429/timeout/5xx (up to 5 attempts); non-429 4xx surface immediately. A claim that still fails is written as a safe not_enough_information fallback row so the batch completes.
- **Determinism:** no `temperature`/`thinking` (rejected on Opus 4.8); all policy is in the deterministic adjudicator, so output is reproducible from the cache.

## 6. Recommendation

Use **`claude-opus-4-8`** for the production run — highest composite (0.682, +0.131 over the next config). The perception/adjudication split means the chosen model can be swapped without touching the policy layer.
