# Wild honey bee habitat suitability — eastern High Atlas (upper Ziz), Morocco

Data and code for the paper *Benchmarking four machine-learning classifiers with spatial
cross-validation for wild honey bee habitat mapping in Morocco*
(L. Naimi, M. Manaouch, Q. B. Pham).

**Version 2 (30 m).** Predictors are extracted on a 30 m grid at the exact positions of the
150 presences and 150 verified absences, with a 3 × 3-cell (90 m) moving window for the
continuous terrain and soil layers. The earlier 300 m version is kept in the git history
(tag `v1-300m`).

## Contents

| Folder | Content |
|---|---|
| `data/training_data_generalised_1km.csv` | 150 presences + 150 verified absences with all model features (30 m / 90 m window). Coordinates are truncated to 1 km (`x_km_1km`, `y_km_1km`, UTM 30N) to protect nest locations |
| `data/*.csv`, `data/best_params.json` | Tuned hyper-parameters, test metrics, spatial-CV results (by fold and summary), permutation importance, SHAP importance, response curves, area per suitability class |
| `rasters/suitability_*_30m.tif` | Suitability maps for RF, XGBoost, SVM, MLP and the ensemble (GeoTIFF, 30 m, EPSG:32630). Values are probability × 100 (0–100, `uint8`); 255 = no data |
| `rasters/suitability_ensemble_classes_30m.tif` | Ensemble in five classes (1 very low … 5 very high), with colour table |
| `scripts/` | Python scripts (predictor stack, dataset, training, evaluation, SHAP, figures, maps) |
| `figures/` | Figures of the paper, including the methodological workflow |

## Grid
30 m cells, 2460 rows × 4150 columns, lower-left corner (266900, 3535978), EPSG:32630;
4,836,384 cells with complete predictors (4,353 km²).

## Main results (30 m)

| Model | Test AUC | Spatial-CV AUC (mean ± SD) |
|---|---|---|
| Random forest | 0.896 | 0.860 ± 0.042 |
| SVM | 0.886 | 0.859 ± 0.070 |
| XGBoost | 0.847 | 0.834 ± 0.066 |
| MLP | 0.787 | 0.826 ± 0.053 |
| Ensemble (mean of 4) | 0.890 | – |

## Field data
Presences and absences were verified between February and July 2026 by field survey and
interviews with 12 shepherds. Exact nest coordinates are not published; they can be
requested from the authors for research purposes.

## Software
Python 3, scikit-learn, XGBoost, SHAP, rasterio, scipy, pandas, numpy, matplotlib.

## Order of the scripts
`build30.py` (30 m predictor stack from the source rasters) → `build_d30.py` (training table,
grid table) → `train.py` → `evaluate.py` → `shap_run.py` → `shap_figs.py` → `curves_maps2.py`
→ `maps_figs.py` → `maps_single_nopts.py` → `panel.py` → `maps_extra.py`; `stats30.py` gives the
study-area statistics. `build30.py`, `build_d30.py` and the map scripts need the source
rasters and the exact coordinates, which are not included; `train.py` and `evaluate.py` can be
run on the generalised training file (all features are complete).

