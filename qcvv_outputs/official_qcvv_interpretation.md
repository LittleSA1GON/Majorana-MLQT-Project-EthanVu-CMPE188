# Official-Style QCVV Interpretation

## Executive conclusion

Current evidence level: **Level 1 - model-assisted characterization**.

The project supports QCVV-style characterization from trained model artifacts. A stronger official claim requires physical calibration, raw or prepared trace records, uncertainty bounds, drift analysis, and predeclared thresholds.

## Claim boundary

### Claims currently supported

- model-assisted QCVV characterization

- effective model-time lifetimes

- CT-HMM-derived transition-rate estimates

- hidden-state prediction confidence and entropy

- pipeline-vs-individual model consistency

- dwell and drift diagnostics from exported predictions


### Claims not yet supported unless the missing data are added

- certified physical lifetime in seconds

- certified parity readout fidelity

- true assignment error without calibration labels

- QND repeatability without repeated-readout or pre/post data

- measurement-induced transition probability without direct experiment alignment

- device-level validation without raw/prepared traces and run metadata

## Missing items for official/certification-style QCVV

- physical timestep dt for parity/x_loop/z_loop

- hidden-state to physical-state labels

- calibration-label readout metrics

- raw/prepared trace overlays or direct raw-data checks


### Direct-data checklist from extraction

| need | why | status | how_to_use |
| --- | --- | --- | --- |
| physical timestep dt for each family | Converts tau_0/tau_1 and dwell-window statistics from model units into seconds. | missing | Pass --dt-json dt.json with keys parity, x_loop, z_loop. |
| calibration labels or known state preparations | Maps hidden state 0/1 to even/odd parity or physical X/Z loop states and enables true assignment error/readout fidelity. | missing | Pass --state-label-json for state names and use qcvv_calibrate.py for confusion matrices/fidelity. |
| raw or prepared I/Q traces with sample_dt and run/time metadata | Needed for physical drift analysis, raw trace dwell-time posteriors, repeated-readout/QND metrics, and model inference on new data. | not available in extracted model artifacts unless prepared bundles are accessible locally | Keep columns such as family, run_id, sequence_id, sample_index, t, I, Q, sample_dt. |
| repeated-readout or pre/post measurement experiments | Needed to quantify measurement backaction and QND repeatability rather than only decoder confidence. | not inferable from model artifacts alone | Prepare paired pre/post or repeated measurements and align them by run/sequence/time. |
| individual CNN-GRU/DMM prediction exports if you want prediction-level comparison for those baselines | The individual bundle has checkpoints and training metrics for some neural models, but no prediction NPZ for CNN-GRU/DMM in the current project files. | missing when prediction_npz is blank in artifact_inventory.csv | Run model inference/export on a shared held-out dataset, then rerun qcvv_extract.py. |

## CT-HMM lifetime interpretation

CT-HMM outputs are the most interpretable lifetime estimates in the current project. They should be described as effective Markovian lifetimes unless sequence-aware dwell, drift, and non-Markovian checks support a stronger model.

| bundle | family | gamma_01 | gamma_10 | tau_0_model_units | tau_1_model_units | tau_mean_model_units | train_log_likelihood_per_obs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | parity | 2.47621 | 4.96012 | 0.403844 | 0.201608 | 0.302726 | -13.4066 |
| pipeline | parity | 2.49686 | 5.01121 | 0.400503 | 0.199552 | 0.300028 | -13.4076 |
| individual | x_loop | 3.552659e+05 | 3.906578e+05 | 2.814793e-06 | 2.559785e-06 | 2.687289e-06 | -11.3288 |
| pipeline | x_loop | 3.555079e+05 | 3.901360e+05 | 2.812877e-06 | 2.563209e-06 | 2.688043e-06 | -11.3295 |
| individual | z_loop | 1795.29 | 2939.4 | 5.570145e-04 | 3.402058e-04 | 4.486102e-04 | -10.5347 |
| pipeline | z_loop | 1789.43 | 2938.97 | 5.588383e-04 | 3.402556e-04 | 4.495470e-04 | -10.5321 |

## Pipeline vs individual interpretation

Pipeline should be treated as the primary QCVV workflow. Individual runs are ablation/baseline evidence. Ratios near 1 indicate stable model-derived quantities across workflows; large deviations indicate sensitivity to training organization or missing prediction exports.

| comparison_type | family | model | metric | pipeline_value | individual_value | absolute_delta_pipeline_minus_individual | ratio_pipeline_over_individual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prediction_metrics | parity | gpu_cthmm | mean_confidence | 0.996908 | 0.996909 | -8.370988e-07 | 0.999999 |
| prediction_metrics | parity | gpu_cthmm | mean_entropy_bits | 0.010646 | 0.01064 | 5.902111e-06 | 1.00055 |
| prediction_metrics | parity | gpu_cthmm | state_0_occupancy | 0.642105 | 0.642105 | 0 | 1 |
| cthmm_lifetimes | parity | gpu_cthmm | tau_mean_model_units | 0.300028 | 0.302726 | -0.00269791 | 0.991088 |
| prediction_metrics | parity | gpu_hsmm_duration | state_0_occupancy | 0.631473 | 0.630764 | 7.090819e-04 | 1.00112 |
| prediction_metrics | x_loop | gpu_cthmm | mean_confidence | 0.60686 | 0.60676 | 1.001634e-04 | 1.00017 |
| prediction_metrics | x_loop | gpu_cthmm | mean_entropy_bits | 0.949358 | 0.949439 | -8.110221e-05 | 0.999915 |
| prediction_metrics | x_loop | gpu_cthmm | state_0_occupancy | 0.508844 | 0.505091 | 0.00375269 | 1.00743 |
| cthmm_lifetimes | x_loop | gpu_cthmm | tau_mean_model_units | 2.688043e-06 | 2.687289e-06 | 7.536319e-10 | 1.00028 |
| prediction_metrics | x_loop | gpu_hsmm_duration | state_0_occupancy | 0.297796 | 0.318495 | -0.0206989 | 0.93501 |
| prediction_metrics | z_loop | gpu_cthmm | mean_confidence | 0.810327 | 0.809275 | 0.00105195 | 1.0013 |
| prediction_metrics | z_loop | gpu_cthmm | mean_entropy_bits | 0.614654 | 0.617635 | -0.0029809 | 0.995174 |
| prediction_metrics | z_loop | gpu_cthmm | state_0_occupancy | 0.60686 | 0.609391 | -0.0025308 | 0.995847 |
| cthmm_lifetimes | z_loop | gpu_cthmm | tau_mean_model_units | 4.495470e-04 | 4.486102e-04 | 9.367972e-07 | 1.00209 |
| prediction_metrics | z_loop | gpu_hsmm_duration | state_0_occupancy | 0.661794 | 0.636202 | 0.0255917 | 1.04023 |

## Model role interpretation

| model_family | official_role | use_for | caution |
| --- | --- | --- | --- |
| CT-HMM | interpretable characterization | transition rates and effective lifetimes | Markovian assumption; needs dwell/drift checks |
| CNN-GRU independent | baseline decoder | ablation and learning-without-teacher comparison | do not use if collapsed or low confidence |
| CNN-GRU teacher-assisted | deployable sequence decoder candidate | fast readout if calibrated | teacher agreement is not physical fidelity |
| HSMM | duration/non-Markovian diagnostic | dwell-time checks | not a substitute for raw-trace lifetime analysis |
| DMM | high-capacity sequence model | model-consistency and flexible inference | watch for overconfidence and teacher imitation |

## Highest-confidence decoder candidates

High confidence should be interpreted as internal model certainty. It becomes readout fidelity only after calibration-label analysis.

| bundle | family | stage | model | mean_confidence | mean_entropy_bits | state_0_occupancy | state_1_occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | parity | cthmm | gpu_cthmm | 0.996909 | 0.01064 | 0.642105 | 0.357895 |
| pipeline | parity | 05_dmm | dmm | 0.999291 | 0.00274044 | 0.636915 | 0.363085 |
| individual | x_loop | cthmm | gpu_cthmm | 0.60676 | 0.949439 | 0.505091 | 0.494909 |
| pipeline | x_loop | 01_cthmm | gpu_cthmm | 0.60686 | 0.949358 | 0.508844 | 0.491156 |
| individual | z_loop | cthmm | gpu_cthmm | 0.809275 | 0.617635 | 0.609391 | 0.390609 |
| pipeline | z_loop | 01_cthmm | gpu_cthmm | 0.810327 | 0.614654 | 0.60686 | 0.39314 |

## Model-readiness scorecard

These scores are artifact/model readiness indicators, not certification scores.

| bundle | family | qcvv_model_readiness_score_0_to_100 |
| --- | --- | --- |
| individual | parity | 99.5549 |
| pipeline | parity | 95.6118 |
| individual | x_loop | 85.2299 |
| pipeline | x_loop | 78.6701 |
| individual | z_loop | 94.2669 |
| pipeline | z_loop | 88.7213 |

## Uncertainty interpretation

Bootstrap outputs were detected. They quantify resampling uncertainty of model summaries. They do not replace physical calibration uncertainty.

| bundle | family | model | metric | bootstrap_median | ci_low_2p5 | ci_high_97p5 | resample_unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline | parity | gpu_cthmm | state_0_occupancy | 0.643372 | 0.618465 | 0.667874 | contiguous_blocks |
| pipeline | parity | gpu_cthmm | state_1_occupancy | 0.358777 | 0.330578 | 0.386556 | contiguous_blocks |
| pipeline | parity | gpu_cthmm | window_switch_rate | 0.325988 | 0.304061 | 0.346996 | contiguous_blocks |
| pipeline | parity | gpu_cthmm | mean_confidence | 0.996907 | 0.996699 | 0.997143 | contiguous_blocks |
| pipeline | parity | gpu_cthmm | mean_entropy_bits | 0.0106773 | 0.00990154 | 0.0114384 | contiguous_blocks |
| pipeline | parity | cnn_gru | state_0_occupancy | 0.999999 | 0.999996 | 1 | contiguous_blocks |
| pipeline | parity | cnn_gru | state_1_occupancy | 1.423560e-06 | 0 | 4.306270e-06 | contiguous_blocks |
| pipeline | parity | cnn_gru | window_switch_rate | 2.848512e-06 | 0 | 1.139405e-05 | contiguous_blocks |
| pipeline | parity | cnn_gru | mean_confidence | 0.508904 | 0.508801 | 0.509002 | contiguous_blocks |
| pipeline | parity | cnn_gru | mean_entropy_bits | 0.999766 | 0.999761 | 0.999771 | contiguous_blocks |
| pipeline | parity | cnn_gru | state_0_occupancy | 0.635556 | 0.592606 | 0.672146 | contiguous_blocks |
| pipeline | parity | cnn_gru | state_1_occupancy | 0.368813 | 0.327116 | 0.411765 | contiguous_blocks |
| pipeline | parity | cnn_gru | window_switch_rate | 0.0989552 | 0.0917254 | 0.106304 | contiguous_blocks |
| pipeline | parity | cnn_gru | mean_confidence | 0.916947 | 0.912095 | 0.922372 | contiguous_blocks |
| pipeline | parity | cnn_gru | mean_entropy_bits | 0.283246 | 0.263211 | 0.300754 | contiguous_blocks |
| pipeline | parity | gpu_hsmm_duration | state_0_occupancy | 0.621422 | 0.45697 | 0.778203 | contiguous_blocks |
| pipeline | parity | gpu_hsmm_duration | state_1_occupancy | 0.36686 | 0.20841 | 0.525719 | contiguous_blocks |
| pipeline | parity | gpu_hsmm_duration | window_switch_rate | 0.0203347 | 0.00137867 | 0.0445362 | contiguous_blocks |
| pipeline | parity | dmm | state_0_occupancy | 0.63881 | 0.596665 | 0.693954 | contiguous_blocks |
| pipeline | parity | dmm | state_1_occupancy | 0.366921 | 0.315061 | 0.422423 | contiguous_blocks |
| pipeline | parity | dmm | window_switch_rate | 0.00212641 | 0.00173766 | 0.00255551 | contiguous_blocks |
| pipeline | parity | dmm | mean_confidence | 0.999299 | 0.99909 | 0.999462 | contiguous_blocks |
| pipeline | parity | dmm | mean_entropy_bits | 0.00271899 | 0.00213303 | 0.00352138 | contiguous_blocks |
| pipeline | x_loop | gpu_cthmm | state_0_occupancy | 0.507733 | 0.485536 | 0.532716 | contiguous_blocks |
| pipeline | x_loop | gpu_cthmm | state_1_occupancy | 0.49128 | 0.467688 | 0.513435 | contiguous_blocks |
| pipeline | x_loop | gpu_cthmm | window_switch_rate | 0.377916 | 0.331106 | 0.415767 | contiguous_blocks |
| pipeline | x_loop | gpu_cthmm | mean_confidence | 0.60681 | 0.604927 | 0.609081 | contiguous_blocks |
| pipeline | x_loop | gpu_cthmm | mean_entropy_bits | 0.949444 | 0.947344 | 0.950913 | contiguous_blocks |
| pipeline | x_loop | cnn_gru | state_0_occupancy | 0.999568 | 0.999275 | 0.999769 | contiguous_blocks |
| pipeline | x_loop | cnn_gru | state_1_occupancy | 4.373068e-04 | 2.360920e-04 | 6.546188e-04 | contiguous_blocks |

_Showing 30 of 93 rows._

## Drift / nonstationarity interpretation

Drift diagnostics were detected. Large relative ranges should be investigated before claiming stationary lifetimes or stable readout performance.

| bundle | family | model | metric | n_bins | mean_across_bins | range_max_minus_min | relative_range_over_mean_abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| individual | parity | gpu_cthmm | mean_confidence | 20 | 0.996909 | 0.00822635 | 0.00825186 |
| individual | parity | gpu_cthmm | mean_entropy_bits | 20 | 0.01064 | 0.0272879 | 2.56464 |
| individual | parity | gpu_cthmm | state_0_occupancy | 20 | 0.642105 | 0.985516 | 1.53482 |
| individual | parity | gpu_cthmm | state_1_occupancy | 20 | 0.357895 | 0.985516 | 2.75365 |
| individual | parity | gpu_cthmm | switch_rate | 20 | 0.0227126 | 0.253414 | 11.1574 |
| individual | parity | gpu_hsmm_duration | state_0_occupancy | 20 | 0.6278 | 1 | 1.59286 |
| individual | parity | gpu_hsmm_duration | state_1_occupancy | 20 | 0.3722 | 1 | 2.68673 |
| individual | parity | gpu_hsmm_duration | switch_rate | 20 | 0.0117483 | 0.204733 | 17.4266 |
| pipeline | parity | cnn_gru | mean_confidence | 20 | 0.508904 | 0.0033395 | 0.00656213 |
| pipeline | parity | cnn_gru | mean_confidence | 20 | 0.917133 | 0.15402 | 0.167936 |
| pipeline | parity | cnn_gru | mean_entropy_bits | 20 | 0.999766 | 1.702401e-04 | 1.702800e-04 |
| pipeline | parity | cnn_gru | mean_entropy_bits | 20 | 0.28233 | 0.494517 | 1.75156 |
| pipeline | parity | cnn_gru | state_0_occupancy | 20 | 0.999999 | 2.851115e-05 | 2.851119e-05 |
| pipeline | parity | cnn_gru | state_0_occupancy | 20 | 0.632015 | 0.919599 | 1.45503 |
| pipeline | parity | cnn_gru | state_1_occupancy | 20 | 1.425557e-06 | 2.851115e-05 | 20 |
| pipeline | parity | cnn_gru | state_1_occupancy | 20 | 0.367985 | 0.919599 | 2.49901 |
| pipeline | parity | cnn_gru | switch_rate | 20 | 2.851196e-06 | 5.702392e-05 | 20 |
| pipeline | parity | cnn_gru | switch_rate | 20 | 0.0987897 | 0.202036 | 2.04511 |
| pipeline | parity | dmm | mean_confidence | 20 | 0.999291 | 0.00651752 | 0.00652214 |
| pipeline | parity | dmm | mean_entropy_bits | 20 | 0.00274044 | 0.0237082 | 8.65124 |
| pipeline | parity | dmm | state_0_occupancy | 20 | 0.636915 | 0.999943 | 1.56998 |
| pipeline | parity | dmm | state_1_occupancy | 20 | 0.363085 | 0.999943 | 2.75402 |
| pipeline | parity | dmm | switch_rate | 20 | 0.00213697 | 0.013144 | 6.15077 |
| pipeline | parity | gpu_cthmm | mean_confidence | 20 | 0.996908 | 0.00822551 | 0.00825102 |
| pipeline | parity | gpu_cthmm | mean_entropy_bits | 20 | 0.010646 | 0.02728 | 2.56248 |
| pipeline | parity | gpu_cthmm | state_0_occupancy | 20 | 0.642105 | 0.985516 | 1.53482 |
| pipeline | parity | gpu_cthmm | state_1_occupancy | 20 | 0.357895 | 0.985516 | 2.75365 |
| pipeline | parity | gpu_cthmm | switch_rate | 20 | 0.0235438 | 0.269752 | 11.4575 |
| pipeline | parity | gpu_hsmm_duration | state_0_occupancy | 20 | 0.627688 | 1 | 1.59315 |
| pipeline | parity | gpu_hsmm_duration | state_1_occupancy | 20 | 0.372312 | 1 | 2.68592 |
| pipeline | parity | gpu_hsmm_duration | switch_rate | 20 | 0.0131886 | 0.228811 | 17.3491 |
| individual | x_loop | gpu_cthmm | mean_confidence | 20 | 0.60676 | 0.0907904 | 0.149632 |
| individual | x_loop | gpu_cthmm | mean_entropy_bits | 20 | 0.949439 | 0.0928111 | 0.0977536 |
| individual | x_loop | gpu_cthmm | state_0_occupancy | 20 | 0.505091 | 0.715376 | 1.41633 |
| individual | x_loop | gpu_cthmm | state_1_occupancy | 20 | 0.494909 | 0.715376 | 1.44547 |
| individual | x_loop | gpu_cthmm | switch_rate | 20 | 0.300737 | 0.317561 | 1.05594 |
| individual | x_loop | gpu_hsmm_duration | state_0_occupancy | 20 | 0.322432 | 0.521452 | 1.61725 |
| individual | x_loop | gpu_hsmm_duration | state_1_occupancy | 20 | 0.677568 | 0.521452 | 0.769593 |
| individual | x_loop | gpu_hsmm_duration | switch_rate | 20 | 0.0875596 | 0.129127 | 1.47473 |
| pipeline | x_loop | cnn_gru | mean_confidence | 20 | 0.504268 | 0.00208598 | 0.00413665 |
| pipeline | x_loop | cnn_gru | mean_confidence | 20 | 0.59407 | 0.0887664 | 0.149421 |
| pipeline | x_loop | cnn_gru | mean_entropy_bits | 20 | 0.99994 | 5.645928e-05 | 5.646269e-05 |
| pipeline | x_loop | cnn_gru | mean_entropy_bits | 20 | 0.960332 | 0.0774663 | 0.0806662 |
| pipeline | x_loop | cnn_gru | state_0_occupancy | 20 | 0.999548 | 0.0044086 | 0.00441059 |
| pipeline | x_loop | cnn_gru | state_0_occupancy | 20 | 0.478957 | 0.716882 | 1.49676 |
| pipeline | x_loop | cnn_gru | state_1_occupancy | 20 | 4.516129e-04 | 0.0044086 | 9.7619 |
| pipeline | x_loop | cnn_gru | state_1_occupancy | 20 | 0.521043 | 0.716882 | 1.37586 |
| pipeline | x_loop | cnn_gru | switch_rate | 20 | 6.559845e-04 | 0.00494677 | 7.54098 |
| pipeline | x_loop | cnn_gru | switch_rate | 20 | 0.0683729 | 0.0823744 | 1.20478 |
| pipeline | x_loop | dmm | mean_confidence | 20 | 0.514256 | 4.318538e-04 | 8.397647e-04 |
| pipeline | x_loop | dmm | mean_entropy_bits | 20 | 0.999397 | 4.042824e-05 | 4.045264e-05 |
| pipeline | x_loop | dmm | state_0_occupancy | 20 | 0 | 0 | 0 |
| pipeline | x_loop | dmm | state_1_occupancy | 20 | 1 | 0 | 0 |
| pipeline | x_loop | dmm | switch_rate | 20 | 0 | 0 | 0 |
| pipeline | x_loop | gpu_cthmm | mean_confidence | 20 | 0.60686 | 0.0894485 | 0.147396 |
| pipeline | x_loop | gpu_cthmm | mean_entropy_bits | 20 | 0.949358 | 0.0914018 | 0.0962775 |
| pipeline | x_loop | gpu_cthmm | state_0_occupancy | 20 | 0.508844 | 0.716559 | 1.40821 |
| pipeline | x_loop | gpu_cthmm | state_1_occupancy | 20 | 0.491156 | 0.716559 | 1.45892 |
| pipeline | x_loop | gpu_cthmm | switch_rate | 20 | 0.316658 | 0.341004 | 1.07689 |
| pipeline | x_loop | gpu_hsmm_duration | state_0_occupancy | 20 | 0.30065 | 0.546341 | 1.8172 |

_Showing 60 of 93 rows._

## Acceptance criteria required before official claim

- Physical dt is defined for parity, x_loop, and z_loop.

- Hidden states 0/1 are mapped to physical parity or loop states using calibration data.

- Readout fidelity/assignment error is computed against physical labels, not another model alone.

- Lifetime estimates include confidence intervals and specify the time unit.

- Dwell-time distributions and drift checks do not contradict the Markovian lifetime interpretation, or the report explicitly calls lifetimes effective Markovian rates.

- Repeated-readout or pre/post records support QND/repeatability and measurement-backaction claims.

- Predeclared thresholds are used for pass/fail/certification-style statements.

## Recommended final wording

> This project implements a model-assisted QCVV workflow for parity readout and tetron X/Z loop lifetime characterization. CT-HMM provides interpretable transition-rate estimates; CNN-GRU and DMM provide learned sequence decoders; HSMM provides dwell-time diagnostics. Pipeline results are the primary workflow, while individual runs serve as independent baseline evidence. Current artifacts support effective model-time lifetimes and internal model-consistency checks. Fully official QCVV claims require physical timing metadata, state-label calibration, raw/prepared traces, repeated-readout data, uncertainty bounds, drift checks, and predeclared acceptance thresholds.
