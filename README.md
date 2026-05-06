# Majorana-MLQT-Project-EthanVu-CMPE188

## ML-Assisted Characterization of Majorana-Inspired Parity and Loop Readout Artifacts

**Author:** Ethan Vu  
**Course:** CMPE 188  
**Project area:** Machine learning for quantum readout characterization, hidden-state modeling, and QCVV-style analysis

![QCVV Model Readiness Scorecard](qcvv_outputs/figures/qcvv_model_readiness_scorecard.png)

---

## Summary

This repository provides a concise machine-learning characterization pipeline for parity, X-loop, and Z-loop readout artifacts. It compares interpretable hidden-state models with neural sequence models to evaluate two-state structure, dwell behavior, model confidence, drift, reproducibility, and agreement across independent workflows.

The purpose of the repository is to turn exported quantum-readout artifacts into clear, auditable model diagnostics that help identify which measurement families are ready for deeper calibration and physical follow-up.

---

## Scope

This project is a software characterization study focused on exported artifacts and model-derived evidence. It supports artifact-level comparison across measurement families using confidence, entropy, transition-rate estimates, model-unit effective lifetimes, dwell summaries, drift checks, bootstrap stability, and model agreement.

Physical interpretation requires calibrated sample timing, calibrated raw traces, prepared-state labels, and repeated-readout experiments. The results are intended as a structured readiness and characterization layer for future validation work.

### QCVV-style repository classification

The table below defines how QCVV-related language is used in this repository.

| QCVV-related category | Role in this repository | Purpose |
|---|---|---|
| Characterization | Primary scope | Estimate model-derived behavior and uncertainty from parity and loop-readout artifacts. |
| Comparative model assessment | Primary scope | Compare multiple model classes on the same measurement families. |
| Internal readiness scoring | Repository metric | Summarize artifact completeness and model consistency for project-level review. |
| Verification preparation | Future-facing use | Identify the additional metadata and calibration structure required for specification-based testing. |
| Validation preparation | Future-facing use | Organize model evidence so calibrated experimental follow-up can target the strongest measurement families. |

The phrase **model-assisted characterization** describes the repository purpose: ML-based analysis of exported parity and loop-readout artifacts.

---

## Project question

Quantum readout artifacts can contain noisy, ambiguous, and model-dependent structure. This repository asks:

> Can hidden-state and neural sequence models characterize parity-style readout artifacts and identify which measurement families show the most consistent two-state behavior?

The analysis centers on three measurement families:

| Measurement family | Role | Main characterization question |
|---|---|---|
| `parity` | General parity-style two-state readout artifact | How clearly does the family support hidden-state structure? |
| `x_loop` | X-loop-style loop readout artifact | How stable and reproducible is the X-loop family across models? |
| `z_loop` | Z-loop-style loop readout artifact | How strongly does the Z-loop family compare with parity and X-loop diagnostics? |


---

## Data provenance and preparation

The data used by this repository was obtained from two public Microsoft quantum-device data repositories and their linked public datasets:

| Source repository | Source paper / dataset role | Data used in this project |
|---|---|---|
| `microsoft/azure-quantum-parity-readout` | Code and data workflow for **Interferometric Single-Shot Parity Measurement in an InAs-Al Hybrid Device**. The Microsoft repository points to measured, converted, and simulated datasets distributed through Zenodo and prepared through its `prepare_data.py` workflow. | Parity-style readout artifacts, especially the MPR-style families used here as `mpr_A1`, `mpr_A2`, and `mpr_B1`. |
| `microsoft/microsoft-quantum-tetron-lifetimes` | Code and data workflow for **Distinct Lifetimes for X and Z Loop Measurements in a Majorana Tetron Device**. The Microsoft repository points to measurement datasets on Zenodo and organizes raw, converted, and processed files through its `prepare_data.py` workflow. | Loop-readout artifacts used here as `xmpr` for X-loop analysis and `zmpr` for Z-loop analysis. |

Source links:

```text
https://github.com/microsoft/azure-quantum-parity-readout
https://github.com/microsoft/microsoft-quantum-tetron-lifetimes
```

In this repository, the public Microsoft data workflows are treated as the upstream data source, and this project adds a machine-learning preparation layer on top of the exported readout artifacts. The Microsoft repositories provide the source measurement organization, CQ-converted readout files, processed datasets, and figure-reproduction notebooks. This repository selects the parity, X-loop, and Z-loop artifacts needed for sequence modeling and converts them into a consistent family-level format.

For the final characterization pass, the source material is organized into three analysis families:

| Prepared family | Upstream origin | Local prepared artifact | Preparation purpose |
|---|---|---|---|
| `parity` | `azure-quantum-parity-readout` MPR-style parity measurements | Combined `mpr_A1`, `mpr_A2`, and `mpr_B1` into `parity_combined.h5`, then into `prepared_parity.pt` | Build a two-state parity-readout artifact family for hidden-state and neural sequence characterization. |
| `x_loop` | `microsoft-quantum-tetron-lifetimes` X-loop measurement workflow | `xmpr_Cq.h5` to `prepared_x_loop.pt` | Prepare X-loop readout traces for uncertainty, drift, dwell, and model-agreement checks. |
| `z_loop` | `microsoft-quantum-tetron-lifetimes` Z-loop measurement workflow | `zmpr_Cq.h5` to `prepared_z_loop.pt` | Prepare Z-loop readout traces for comparison against parity and X-loop diagnostics. |

The preparation workflow is intentionally staged so each output has a clear lineage:

1. **Download or stage upstream data** from the two Microsoft repositories and their linked Zenodo datasets. The upstream scripts organize raw, converted, simulated, and processed datasets into reproducible data folders.
2. **Select readout artifacts** that match the repository scope: MPR-style parity measurements from `azure-quantum-parity-readout`, plus `xmpr` and `zmpr` loop-readout artifacts from `microsoft-quantum-tetron-lifetimes`.
3. **Use CQ-converted readout channels** where available. The prepared sequence traces use the `Cq` and `iCq` channels so each sample carries a two-channel readout trajectory.
4. **Consolidate parity runs** with `src/data_management/parity-combine.py`. The parity preparation combines `mpr_A1`, `mpr_A2`, and `mpr_B1` into a single family-level HDF5 container while preserving run names, source-file paths, root attributes, and creation metadata.
5. **Preprocess each family** with `src/data_management/preprocess.py`. This step standardizes parity, X-loop, and Z-loop artifacts into prepared HDF5 outputs, extracts arrays, records coordinate metadata, builds sequence windows, and writes confirmation reports for auditability.
6. **Convert to model-ready bundles** with `src/data_management/convert.py`. The prepared HDF5 files become PyTorch `.pt` bundles that retain the original source H5 path, file attributes, dataset names, shapes, source dtypes, converted tensor dtypes, run names, coordinate fields, trace channels, and valid sequence lengths.
7. **Audit prepared tensors** using the included audit report. The audited bundles contain `prepared_parity.pt`, `prepared_x_loop.pt`, and `prepared_z_loop.pt`, each with trace arrays shaped as 512-sample, two-channel sequences.
8. **Run model characterization** through the primary `pipeline` workflow and the independent `individual` workflow. The resulting predictions, agreements, dwell summaries, drift checks, bootstrap intervals, and readiness scorecards are collected under `qcvv_outputs/`.

The prepared bundle audit gives the final data layout used by the model pipeline:

| Prepared bundle | Samples | Source runs / files | Trace format | Sample timing retained in bundle |
|---|---:|---|---|---:|
| `prepared_parity.pt` | 701,480 | `mpr_A1`, `mpr_A2`, `mpr_B1` from `parity_combined.h5` | `[512, 2]` traces with `Cq` and `iCq` channels | `9.102834883378819e-05` |
| `prepared_x_loop.pt` | 186,000 | `xmpr_Cq.h5` | `[512, 2]` traces with `Cq` and `iCq` channels | `9.999999974752427e-07` |
| `prepared_z_loop.pt` | 15,015 | `zmpr_Cq.h5` | `[512, 2]` traces with `Cq` and `iCq` channels | `4.999999873689376e-05` |

This data flow keeps the repository focused: public Microsoft quantum-readout datasets become selected family artifacts, selected artifacts become standardized sequence bundles, and sequence bundles become concise QCVV-style characterization summaries.

---

## Repository contents

| Path | Purpose |
|---|---|
| `pipeline/pipeline/` | Primary workflow outputs for parity, X-loop, and Z-loop families. |
| `individual/individual/` | Independent baseline outputs for reproducibility checks. |
| `qcvv_outputs/` | Final scorecards, prediction metrics, agreement tables, dwell metrics, drift summaries, bootstrap outputs, reports, and figures. |
| `qcvv_outputs/figures/` | Visual summaries for readiness, prediction confidence, model agreement, lifetimes, dwell behavior, drift, and bootstrap results. |
| `src/data_management/` | Data consolidation, preprocessing, HDF5-to-bundle conversion, audit, and cleanup utilities. |
| `src/test.py` | Basic repository test and validation script. |
| `README.md` | Project overview and final interpretation. |

Key generated outputs:

- Prepared family HDF5 artifacts from the local preprocessing workspace
- PyTorch-ready `.pt` bundles from prepared HDF5 files
- `qcvv_outputs/qcvv_scorecard.csv`
- `qcvv_outputs/family_summary.csv`
- `qcvv_outputs/prediction_metrics.csv`
- `qcvv_outputs/model_agreements.csv`
- `qcvv_outputs/cross_bundle_model_agreements.csv`
- `qcvv_outputs/cthmm_lifetimes.csv`
- `qcvv_outputs/dwell_metrics.csv`
- `qcvv_outputs/drift_summary.csv`
- `qcvv_outputs/bootstrap_prediction_metrics.csv`
- `qcvv_outputs/bootstrap_model_agreements.csv`
- `qcvv_outputs/bootstrap_dwell_metrics.csv`
- `qcvv_outputs/qcvv_report.md`
- `qcvv_outputs/official_qcvv_interpretation.md`

---

## Model comparison framework

The technical center of the repository is a model-comparison framework. Each model contributes a specific diagnostic view of the same readout artifacts.

| Model | Role | Purposeful contribution |
|---|---|---|
| CT-HMM | Interpretable continuous-time hidden-state model | Estimates hidden-state assignments, transition rates, confidence, entropy, and model-unit effective lifetimes. |
| HSMM | Duration-aware hidden-state model | Tests dwell-time structure and state persistence. |
| CNN-GRU | Independent neural sequence decoder | Tests whether temporal structure can be learned directly from sequence artifacts. |
| Teacher-assisted CNN-GRU | Distilled neural sequence decoder | Tests whether CT-HMM-derived structure transfers into a neural decoder. |
| DMM | Flexible deep temporal model | Provides a higher-capacity temporal cross-check against simpler model conclusions. |

### Why the comparison matters

- **CT-HMM** provides interpretable state-switching diagnostics.
- **HSMM** adds duration-aware dwell behavior.
- **CNN-GRU** supplies an independent neural decoding baseline.
- **Teacher-assisted CNN-GRU** tests transferability of CT-HMM-derived structure.
- **DMM** adds a flexible temporal modeling cross-check.

Agreement across models strengthens the artifact-level characterization. Differences across models reveal uncertainty, instability, or family-specific model dependence that can guide future calibration work.

---

## Characterization metrics

| Metric | Meaning | Interpretation |
|---|---|---|
| Mean confidence | Average model certainty in predicted state assignments | Higher values indicate clearer model state assignments. |
| Entropy | Uncertainty in state probabilities | Lower entropy indicates cleaner two-state separation. |
| State occupancy | Fraction of samples assigned to each hidden state | Shows whether both model states are used meaningfully. |
| Within-bundle model agreement | Agreement among models within the same workflow bundle | Tests consistency across model classes. |
| Cross-bundle same-model agreement | Agreement between pipeline and independent runs for the same model family | Tests reproducibility across workflow variants. |
| CT-HMM transition rates | Estimated hidden-state switching rates | Supports model-unit switching comparisons across families. |
| CT-HMM effective lifetimes | Effective hidden-state persistence computed from transition estimates | Supports model-unit persistence comparisons across families. |
| HSMM dwell metrics | Duration-aware state persistence summaries | Tests whether state occupancy is temporally stable. |
| Drift metrics | Windowed changes in model-derived summaries | Detects changes across the artifact sequence. |
| Bootstrap intervals | Resampling-based uncertainty summaries | Tests summary stability under resampling. |
| Readiness score | Repository-defined artifact and model consistency score | Provides a concise project-level comparison metric. |

---

## Current findings

The repository includes a completed characterization pass for parity, X-loop, and Z-loop artifacts.

### Model-readiness scorecard

| Bundle | Family | Readiness score | Interpretation |
|---|---:|---:|---|
| pipeline | parity | 95.6 / 100 | Strongest primary workflow result. |
| pipeline | x_loop | 78.7 / 100 | Processable family with high uncertainty. |
| pipeline | z_loop | 88.7 / 100 | Moderate-to-strong primary workflow result. |
| individual | parity | 99.6 / 100 | Strongest independent baseline. |
| individual | x_loop | 85.2 / 100 | Improved independent baseline with remaining uncertainty. |
| individual | z_loop | 94.3 / 100 | Strong independent baseline. |

### Family-level comparison

| Family | CT-HMM confidence | CT-HMM entropy | Within-bundle agreement | Characterization result |
|---|---:|---:|---:|---|
| `parity` | 0.9969 | 0.0106 bits | 0.825 | Cleanest two-state structure and strongest overall characterization. |
| `x_loop` | 0.6069 | 0.9494 bits | 0.540 | Highest-uncertainty family and useful stress test for model dependence. |
| `z_loop` | 0.8103 | 0.6147 bits | 0.739 | Stable intermediate result with promising reproducibility. |

---

## Measurement-family interpretation

### Parity

Parity is the strongest family in the current analysis. It has high CT-HMM confidence, low entropy, strong model consistency, and high reproducibility across workflow bundles.

**Interpretation:** The parity artifacts contain the clearest model-derived two-state structure among the analyzed families.

### X-loop

X-loop is the highest-uncertainty family. Its lower confidence, entropy near one bit, and weaker model agreement make it a useful diagnostic case for stress-testing the pipeline.

**Interpretation:** X-loop remains processable and informative, especially for studying ambiguity, model dependence, and artifact-level uncertainty.

### Z-loop

Z-loop is stronger than X-loop and shows good reproducibility. Its confidence, entropy, and agreement metrics place it between parity and X-loop under the current diagnostics.

**Interpretation:** Z-loop is a promising loop-readout family for deeper calibration and timing-aware follow-up.

---

## Recommended figures

1. `qcvv_outputs/figures/qcvv_model_readiness_scorecard.png`
2. `qcvv_outputs/figures/parity_prediction_confidence_all_bundles.png`
3. `qcvv_outputs/figures/x_loop_prediction_confidence_all_bundles.png`
4. `qcvv_outputs/figures/z_loop_prediction_confidence_all_bundles.png`
5. `qcvv_outputs/figures/parity_pipeline_model_agreement_heatmap.png`
6. `qcvv_outputs/figures/x_loop_pipeline_model_agreement_heatmap.png`
7. `qcvv_outputs/figures/z_loop_pipeline_model_agreement_heatmap.png`
8. `qcvv_outputs/figures/cross_bundle_same_model_agreement.png`
9. `qcvv_outputs/figures/cthmm_lifetimes_pipeline_vs_individual.png`
10. `qcvv_outputs/figures/bootstrap_model_agreement.png`

---

## Requirements for deeper physical interpretation

The current repository is designed to support model-assisted artifact characterization. Deeper physical interpretation requires:

- Calibrated sample timestep metadata for converting model-unit rates and lifetimes into physical units.
- Known prepared-state labels for assignment error, confusion matrices, and physical readout fidelity.
- Repeated-readout experiments for QND-style repeatability analysis.
- Calibrated raw traces and device metadata for direct physical validation.

These requirements define the natural next stage after the current model-readiness analysis.

---

## How to inspect the project

Install common Python analysis dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch h5py
```

The main data-preparation utilities are:

```text
src/data_management/parity-combine.py
src/data_management/preprocess.py
src/data_management/convert.py
```

Start with the generated summaries:

```text
qcvv_outputs/qcvv_report.md
qcvv_outputs/official_qcvv_interpretation.md
qcvv_outputs/qcvv_scorecard.csv
qcvv_outputs/family_summary.csv
qcvv_outputs/figures/qcvv_model_readiness_scorecard.png
```

---

## References

- NIST: Classifying single-qubit noise using machine learning  
  https://www.nist.gov/publications/classifying-single-qubit-noise-using-machine-learning
- NIST: Quantum Characterization  
  https://www.nist.gov/programs-projects/quantum-characterization
- Quantum Characterization, Verification, and Validation  
  https://arxiv.org/abs/2503.16383
- A Practical Introduction to Benchmarking and Characterization of Quantum Computers  
  https://arxiv.org/abs/2408.12064
- Microsoft Azure Quantum parity-readout source repository  
  https://github.com/microsoft/azure-quantum-parity-readout
- Microsoft quantum tetron lifetimes source repository  
  https://github.com/microsoft/microsoft-quantum-tetron-lifetimes
- Single-shot parity readout of a minimal Kitaev chain  
  https://www.nature.com/articles/s41586-025-09927-7
- Distinct Lifetimes for X and Z Loop Measurements in a Majorana Tetron Device  
  https://arxiv.org/abs/2507.08795

---

## Final takeaway

This repository is best understood as a comparative characterization pipeline for parity and loop-readout artifacts. Its strongest evidence comes from consistency across CT-HMM, HSMM, CNN-GRU, teacher-assisted CNN-GRU, and DMM outputs.

> **Parity is the cleanest family, Z-loop is promising, and X-loop is the highest-uncertainty diagnostic case.**

The results are model-derived and artifact-level. They provide a concise basis for deciding which measurement families deserve deeper calibration and physical validation.