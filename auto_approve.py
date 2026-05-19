# auto_approve.py
# Author: Nalan-Z
#
# Post-processing filter: removes repeat cases, missing-value cases, and
# out-of-range cases from ML output, then writes an auto-approvable case list
# with upload-ready P3 values.
#
# Requires: config.yaml in the working directory.
# Must be run AFTER Machine_learning.py.
# Usage: python auto_approve.py <batch_number>

from __future__ import division
import time
start_time = time.time()

import os, sys, csv, shutil, argparse
import numpy, pandas
import yaml

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open("config.yaml", "r") as _f:
    _cfg = yaml.safe_load(_f)

RESULTS_DIR       = _cfg["paths"]["results_dir"]
REPEATS_DIR       = _cfg["paths"]["repeats_dir"]
AUTO_APPROVE_LOG  = os.path.join(_cfg["paths"]["log_dir"], _cfg["files"]["auto_approve_log"])





# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog='auto_approve.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''\
Filters ML-processed batch results to produce an auto-approvable case list.

Removes: repeat cases, missing-value cases, outlier plate counts, age out of
range, and any cases where 2b exceeds 2a or 100% Plate 1.

Output: <batch>_auto_approve.xlsx with two sheets:
  Cases      – readable summary of approved cases
  P3 Values  – upload-ready P3 values

Run Machine_learning.py first to generate the processed summary.
''')
parser.add_argument('Accession_Number', metavar='Accession_No.',
                    help='Batch accession number to process.')
args = parser.parse_args()




# ---------------------------------------------------------------------------
# Load processed summary (sheet index 3 = 4th sheet = Summary_for_AA)
# ---------------------------------------------------------------------------
base = str(sys.argv[1]) + '.txt'
infile_ps = base.replace('.txt', '_processed_summary.xlsx')
print(infile_ps)

df = pandas.read_excel(infile_ps, sheet_name=3).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Load repeat file
# ---------------------------------------------------------------------------
repeat_filename = f"Repeats {sys.argv[1]}.xlsx"
repeat_path = os.path.join(REPEATS_DIR, repeat_filename)
rep_df = pandas.read_excel(repeat_path, header=1)
print(repeat_filename)

# Keep only rows above the "Bombs, Rejects" section (if present)
if rep_df['Comments'].str.contains('Bombs, Rejects').any():
    cut_row = rep_df['Comments'].str.contains("Bombs, Rejects")
    row = cut_row[cut_row].index[0]
    rep = rep_df.drop(rep_df.index[row:])
else:
    rep = rep_df
rep = rep.reset_index(drop=True)




# ---------------------------------------------------------------------------
# Load missing-values file
# ---------------------------------------------------------------------------
infile_missing = base.replace('.txt', '_missing_values.xlsx')
infile_missing_path = os.path.join(RESULTS_DIR, infile_missing)
missing = pandas.read_excel(infile_missing_path).reset_index(drop=True)
print(infile_missing)

# ---------------------------------------------------------------------------
# Compute plate difference columns
# ---------------------------------------------------------------------------
df['diff']  = df['P1_raw'] - df['2b_raw']   # should be > 0
df['diff2'] = df['2a_raw']           - df['2b_raw']   # should be > 0

# ---------------------------------------------------------------------------
# Plate count thresholds (±1 SD around batch mean)
# ---------------------------------------------------------------------------
Pl1_min = df['P1_raw'].mean() - df['P1_raw'].std()
Pl1_max = df['P1_raw'].mean() + df['P1_raw'].std()
Pl2_min = df['P2a_raw'].mean() - df['P2a_raw'].std()
Pl2_max = df['P2a_raw'].mean() + df['P2a_raw'].std()

print(f"Plate 1: {Pl1_min:.0f} – {df['P1_raw'].mean():.0f} – {Pl1_max:.0f}")
print(f"Plate 2a: {Pl2_min:.0f} – {df['2a_raw'].mean():.0f} – {Pl2_max:.0f}")

# ---------------------------------------------------------------------------
# Main quality filter
# ---------------------------------------------------------------------------
filtered_list = df[
    (df['overall_validation'] < 1.2) &
    (df['Deficiencies'] < 6) &
    (df['P1_raw'] > Pl1_min) & (df['P1_raw'] < Pl1_max) &
    (df['P2a_raw'] > Pl2_min)           & (df['P2a_raw'] < Pl2_max) &
    (df['Age_raw'] > 14)               & (df['Age_raw'] < 80) 
]

# ---------------------------------------------------------------------------
# Remove repeat and missing cases
# ---------------------------------------------------------------------------
rep2     = pandas.DataFrame(rep['Access Num'].dropna().values, columns=['Accession'])
missing2 = pandas.DataFrame(missing['Accession'].values,       columns=['Accession'])

excluded = (missing2.append(rep2, sort=True)
              .drop_duplicates(subset=['Accession'])
              .reset_index(drop=True))

merged = (filtered_list.append(excluded, sort=True)
            .drop_duplicates(subset=['Accession'], keep=False))
final  = merged.dropna(subset=['overall_validation'])

# ---------------------------------------------------------------------------
# Summary columns
# ---------------------------------------------------------------------------
final_cases = final[[
    "Accession","Age_raw","overall_validation","Deficiencies","Defs",...
]].copy()
final_cases['P1_raw'] = final_cases['P1_raw'].round()
final_cases['P2a_raw']           = final_cases['P2a_raw'].round()
final_cases['P2b_raw']           = final_cases['P2b_raw'].round()
final_cases.overall_validation  = final_cases.overall_validation.round(2)




# ---------------------------------------------------------------------------
# Upload-ready P3 values (dummy columns required by upload template)
# ---------------------------------------------------------------------------
final = (final
         .assign(Cysteine_1=1, Cysteine_3=1, Spectrox_1=95,
                 Low_GLC=86))  # 86 → machine-generated, ready for approval

P3_COLS = [
    "Batch","Accession","B1","B2","B3","B6","B12",...
]
final_p3s = final[P3_COLS]

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
c1  = final_cases['Accession'].count()
c2  = df['Accession'].count()
c3  = c1 / c2 * 100
print('*******')
print(f"Auto-approvable: {c3:.1f}% ({c1} of {c2} cases)")
print('*******')

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
outfile = base.replace('.txt', '_auto_approve.xlsx')
with pandas.ExcelWriter(outfile) as writer:
    final_cases.to_excel(writer, sheet_name='Cases',    index=False)
    final_p3s.to_excel(  writer, sheet_name='P3 Values',index=False)

# ---------------------------------------------------------------------------
# Append to auto-approve log
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(AUTO_APPROVE_LOG), exist_ok=True)
with open(AUTO_APPROVE_LOG, 'a', newline='') as fh:
    csv.writer(fh).writerow([sys.argv[1], c3, df["Accession"].count()])

# ---------------------------------------------------------------------------
# Move repeat files to results directory
# ---------------------------------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
for fname in os.listdir("."):
    if fname.endswith("_repeats.xlsx"):
        shutil.move(fname, os.path.join(RESULTS_DIR, fname))

print(f"\nDone. Runtime: {time.time() - start_time:.1f}s")
