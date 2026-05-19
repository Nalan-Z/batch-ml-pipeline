# learn_dict.py
# Maps each analyte name to the feature columns used for training.
# Keys match the analyte names in ANALYTE_COLS; values are lists of z-score
# column names from the historical dataset that the model trains on.
#
# Populate this with your own feature definitions before running the pipeline.

learn = {
    # Example structure — replace with actual feature columns:
    # "B1": ["Age",  "B2", "B3", ..., "B1"],  # last col = target
}
