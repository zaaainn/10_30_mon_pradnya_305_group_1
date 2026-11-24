# Testing, Validation and Iteration Report

**Original evaluation:** 8 November 2026 · **Severity-model update:** 24 November 2026
**Prototype:** WNTR-generated irrigation readings, Random Forest condition/location/severity models, schedule-aware threshold baseline
**Training and evaluation:** `.venv\\Scripts\\python.exe train_model.py`, then `.venv\\Scripts\\python.exe evaluate_methods.py`

## Scope and procedure

Both methods were applied to the same group of 114 night-schedule held-out records and 19 scenarios (normal operation, three leak and three clog conditions for each zone). The entire night schedule is not included in the training split for the Random Forest. The scenario labels and locations recorded were compared with predictions. All readings are synthetic simulation outputs, not actual performance in the live UOW infrastructure or actual field performance.

## Main results

**Key finding:** On the held-out night schedule, Random Forest achieved higher overall classification accuracy and macro F1 than the schedule-aware threshold baseline. It also detected every simulated fault case, had higher overall leak recall and higher clog recall at each reported severity. The added Random Forest severity classifier achieved 87.0% exact severity accuracy (macro F1 86.9%) across the 108 held-out fault records. This supports Random Forest as the stronger overall classifier for these simulated cases. The threshold baseline was better at locating faults and avoiding false warnings, and does not estimate severity.

| Measure | Random Forest | Schedule-aware threshold |
|---|---:|---:|
| Classification accuracy | 85.1% | 71.1% |
| Macro F1 | 79.4% | 77.5% |
| Fault-location accuracy | 77.8% | 95.4% |
| Missed-fault rate | 0.0% (0/108) | 4.6% (5/108) |
| False-warning rate on normal cases | 50.0% (3/6) | 0.0% (0/6) |
| Mild-clog recall | 61.1% (11/18) | 38.9% (7/18) |
| Moderate-clog recall | 66.7% (12/18) | 55.6% (10/18) |
| Severe-clog recall | 94.4% (17/18) | 66.7% (12/18) |
| Severity exact accuracy on held-out fault records | 87.0% (94/108) | Not estimated |
| Severity macro F1 on held-out fault records | 86.9% | Not estimated |

The severity classifier predicts mild, moderate, or severe for fault cases. Its test covers 108 held-out night fault records, with 36 records per severity. Recall was 100.0% for mild, 75.0% for moderate, and 86.1% for severe. Severity accuracy is a Random Forest capability; the threshold baseline has no severity output. These are simulated labels and readings, not measured field fault severities.

The fault missed-rate is the number of faults that the method did not produce an alert. Even if a wrong prediction of the fault type is detected, the event will still be reported as an alert and thus classification errors will be reported separately with precision, recall and F1.

The most relevant overall measures for the research question are the classification accuracy and macro F1: both are higher for Random Forest in this held-out test. The measures of fault location and false warning show the utility and accuracy of the simpler threshold baseline. The measurements do not show that either method will be the same when applied to field data.

## Test cases: AI-suggested and manually selected

| Test origin and case | Records | Random Forest result | Threshold result | Finding |
|---|---:|---|---|---|
| AI-suggested edge case: mild clogs across Zones A, B and C during the night schedule | 18 | Correct clog class: 61.1%; correct location: 66.7% | Correct clog class: 38.9%; correct location: 72.2%; 27.8% not alerted | Mild clogs are difficult for both methods. The classifier recognized more clogs, while the threshold method located the affected zone more often. |
| Manually selected routine check: normal night operation | 6 | 50.0% classified normal; 50.0% false warnings | 100.0% classified normal; 0.0% false warnings | The classifier raised three false warnings in six normal readings. |
| Manually selected clear-fault check: severe leaks across Zones A, B and C during the night schedule | 18 | Correct leak class and location: 100% | Correct leak class and location: 100% | Both methods handled these clear simulated leak cases. |

The night profile uses a demand multiplier of 0.15. A zero-demand/no-watering fault case is not present in this dataset, so that edge case remains untested.

## Retest and iteration record

Before the severity-model update, the full condition/location comparison was run twice and produced the same prediction CSV byte-for-byte (SHA-256: `4B230141170BA63073541231C974A68CCDCCAD5408932A44786AF4CAA2D38C9D`). The updated evaluation on 24 November includes the new severity predictions; repeat-run reproducibility for this expanded output has not yet been checked. Processing times vary between runs, so they are backend timing observations rather than stable dashboard response-time results.

**Successes:** the evaluation completed for both methods on identical held-out cases; the classifier had higher macro F1 and no missed fault alerts; the threshold method had higher fault-location accuracy and no false warnings; and the Random Forest severity model was evaluated on the same held-out fault records.

**Failures and limitations:** Both methods had low mild-clog recall with the classifier producing three false warnings and the threshold method having poor location accuracy and no zero-demand test. Currently, the response-time metric only accounts for the backend processing time, and not the complete time from the user clicking on a dashboard scenario until he or she sees the result.

**Iteration decision:** Do not tune model or threshold settings on the held-out night results and report this set separately. For the next iteration, include explicit zero demand conditions, examine normal operation false warning and mild clog errors on train/valid data, and test a new version on a new (held-out) test set. Measure end to end response time in the browser independently.

## Dashboard update and retest — 9 November 2026

The Study area page now displays the geographic context image and a separate schematic 3 km boundary, displacing the pipeline overlay, and without the schedule/hour selectors. The simulated source, manifold, zones, readings and scenario controls are still found in the pipeline view. Method comparison now identifies the measured advantages of the Random Forest while retaining the location and false-warning advantages of the baseline, and suggests some links with this report. The results on the page represent the held-out evaluation described above (but no results or source readings were changed).

When added the page separation and report link, the local dataset and dashboard validation suite passed **8 tests in 9.975 seconds**. This includes pipeline overlay checks that Study area does not contain a pipeline overlay, pipeline views rendering pipeline network controls, and Method comparison rendering and report URL returning current report. This is a local verification only, the hosted Render deploy is still to finish successfully.

## Evidence files

- `outputs/method_comparison_metrics.json` — machine-readable metric summary.
- `outputs/method_comparison_predictions.csv` — per-record truth and both methods' predictions.
