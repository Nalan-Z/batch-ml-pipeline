# predict_dict.py
# Maps each analyte name to the feature columns used for prediction.
# Keys match the analyte names in ANALYTE_COLS; values are lists of z-score
# column names from the processed CSV that the model predicts on.
#
# Populate this with your own feature definitions before running the pipeline.

predict = {
    # Example structure — replace with actual feature columns:
    # "B1": ["Age", "B1", "B2", "B3", ...],
}
