#!/usr/bin/env python3

#-----------------------------------------------------------------------------
# Find differences between two csv dataframe files over common columns
#
# USAGE   : ./diff.py data_file1.csv data_file2.csv
#
# Outputs : min/max/avg/std_dev  of common columns
#------------------------------------------------------------------------------


import pandas as pd
import sys

# ==========================================
# CONFIGURATION
# ==========================================

if len(sys.argv) == 3:
    DATA_FILE1 = sys.argv[1]
    DATA_FILE2 = sys.argv[2]
else:
    DATA_FILE1 = 'prot_self.energy.csv'
    DATA_FILE2 = 'prot_self.energy2.csv'

# Character used to denote comment lines at the top of the file
COMMENT_CHAR = '#'

# Delimiter: r'\s+' matches any number of whitespace characters (spaces, tabs)
# To use a standard comma-separated file, change this to ','
DELIMITER = r'\s+' 

# Determines whether to compute the difference as (File1 - File2) 
# or use Absolute Differences abs(File1 - File2)
USE_ABSOLUTE_DIFFERENCE = True
# ==========================================


def main():
    print(f"Loading '{DATA_FILE1}' and '{DATA_FILE2}'...")
    
    try:
        df1 = pd.read_csv(DATA_FILE1, comment=COMMENT_CHAR, sep=DELIMITER, header="infer")
        df2 = pd.read_csv(DATA_FILE2, comment=COMMENT_CHAR, sep=DELIMITER, header="infer")
    except FileNotFoundError as e:
        print(f"Error: Could not find the file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    # Find common columns based on headers
    common_cols = list(set(df1.columns).intersection(set(df2.columns)))
    
    if not common_cols:
        print("No common columns found between the two files.")
        sys.exit(0)

    # Filter to only include numeric columns (math operations require numbers)
    numeric_cols = [
        col for col in common_cols 
        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col])
    ]

    if not numeric_cols:
        print(f"Found common columns ({common_cols}), but none are purely numeric.")
        sys.exit(0)

    # Align dataframes if they have a different number of rows
    min_rows = min(len(df1), len(df2))
    if len(df1) != len(df2):
        print(f"Warning: Row counts differ ({len(df1)} vs {len(df2)}). Comparing the first {min_rows} rows.")
    
    df1_aligned = df1.iloc[:min_rows]
    df2_aligned = df2.iloc[:min_rows]

    print(f"\nComparing {len(numeric_cols)} common numeric columns...\n")
    print("-" * 77)
    print(f"{'Column Name':<20} | {'Min Diff':<12} | {'Max Diff':<12} | {'Avg Diff':<12} | {'Diff Std Dev':<12}")
    print("-" * 77)


    # Find indices of common columns in file 1 to retain proper order in output
    df1_cols = df1.columns.tolist()
    indices = []

    for col in numeric_cols:
        idx = df1_cols.index(col)
        if idx == -1:
            print(f"ERROR: Could not find index of common column {col}")
            sys.exit(1)
        indices.append(idx)


    # Calculate differences and print results
    for i in sorted(indices):
        col = df1_cols[i]

        # Calculate row-by-row difference
        diff = df1_aligned[col] - df2_aligned[col]
        
        if USE_ABSOLUTE_DIFFERENCE:
            diff = diff.abs()

        min_diff = diff.min()
        max_diff = diff.max()
        avg_diff = diff.mean()
        std_diff = diff.std()

        print(f"{col:<20} | {min_diff:<12.4f} | {max_diff:<12.4f} | {avg_diff:<12.4f} | {std_diff:<12.4f}")
        
    print("-" * 77)

if __name__ == "__main__":
    main()
