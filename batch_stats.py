# batch_stats.py
#
# Compute descriptive statistics (min, max, mean, std, count) for P1 analyte
# values from a preprocessed batch file and write them to an Excel summary.
#
# Usage: python batch_stats.py <batch>.xlsx

import time
start_time = time.time()

import os, sys
import pandas

p1s = sys.argv[1]
df  = pandas.read_excel(p1s).reset_index(drop=True)

df = df.drop(columns=['Batch', 'Accession', 'Gender'], errors='ignore')

ANALYTE_COLS = ['Vit A', 'B1', 'B12', 'B2', 'B3', 'B6', 'VitC', 'Vit D3', ...
]
df = df[[c for c in ANALYTE_COLS if c in df.columns]]

df_stats = (df.describe()
              .drop(index=['25%', '50%', '75%'])
              .reindex(['min', 'max', 'mean', 'std', 'count']))
trans = df_stats.T

outfile = sys.argv[1].replace('_P1.xlsx', '_batch_stats.xlsx')
trans.to_excel(outfile)

print(f"Done. Runtime: {time.time() - start_time:.1f}s")
