#!/usr/bin/env python3

import os
import glob
import shutil

#----------------------------------------------------------
# Simple script to drop all zero-columns from data files
#----------------------------------------------------------

INPUT_FILES: list[str] = glob.glob('*.energy.csv')     # list of input data files

ZERO_THRESHOLD: float = 0.03      # a value is considered 0 if abs(value) <= threshold.
                                  # set exactly to 0.0 to only drop exact zeroes

OVERWRITE = True
OUTPUT_SUFFIX = '.filtered'     # only when OVERWRITE is False

COMMENT_TOKEN = '#'
OUTPUT_DELIMITER = ' '         



# ------------------------------
# MAIN
#-------------------------------

# Terminal colors for pretty logging
class LogStyle:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def log_info(msg):
    print(f"{LogStyle.CYAN}{LogStyle.BOLD}[INFO]{LogStyle.RESET} {msg}")

def log_success(msg):
    print(f"{LogStyle.GREEN}{LogStyle.BOLD}[SUCCESS]{LogStyle.RESET} {msg}")

def log_warn(msg):
    print(f"{LogStyle.YELLOW}{LogStyle.BOLD}[WARN]{LogStyle.RESET} {msg}")

def process_csv(in_path, out_path) -> bool:
    comments = []
    data = []

    if not os.path.exists(in_path):
        log_warn(f"'{in_path}': File Not Found")
        return False

    # 1. Read file and separate comments from data
    with open(in_path, 'r') as f:
        for line in f:
            if line.startswith(COMMENT_TOKEN):
                comments.append(line)
            else:
                line_stripped = line.strip()
                if line_stripped:
                    # Splits by arbitrary whitespace if no delimiter is provided to split()
                    data.append(line_stripped.split())

    if not data:
        log_warn(f"'{in_path}': No data rows found.")
        return False

    # 2. Identify which columns to keep
    # Get the max number of columns in case rows are jagged
    max_cols = max(len(row) for row in data)
    cols_to_keep = []

    for col_idx in range(max_cols):
        has_numeric = False
        has_non_zero = False

        for row in data:
            if col_idx < len(row):
                val = row[col_idx]
                try:
                    f_val = float(val)
                    has_numeric = True
                    # If any numeric value is not mathematically zero, we keep the column
                    if abs(f_val) > ZERO_THRESHOLD:
                        has_non_zero = True
                except ValueError:
                    # Ignore headers or string data. We don't fail here.
                    pass

        # KEEP logic:
        # - If it has non-zero numbers, keep it.
        # - If it has NO numbers at all (purely string categories/text), keep it.
        # Drop ONLY if it has numbers and ALL numbers evaluate to 0.0
        if has_non_zero or not has_numeric:
            cols_to_keep.append(col_idx)

    dropped = max_cols - len(cols_to_keep)
    if dropped == 0:
        log_warn(f"{LogStyle.YELLOW}'{in_path}': No columns dropped{LogStyle.RESET}")
        return False

    # 3. Write out to new file
    with open(out_path, 'w') as f:
        # Write preserved comments
        for comment in comments:
            f.write(comment)

        # Write filtered data rows
        for row in data:
            filtered_row = [row[i] for i in cols_to_keep if i < len(row)]
            f.write(OUTPUT_DELIMITER.join(filtered_row) + '\n')

    # Calculate stats for logging
    drop_msg = f"{LogStyle.RED}Dropped {dropped}{LogStyle.RESET}"

    log_success(f"Processed {LogStyle.BOLD}'{in_path}'{LogStyle.RESET} "
                f"-> Kept {len(cols_to_keep)}/{max_cols} cols ({drop_msg}) "
                f"-> Saved as {LogStyle.BOLD}'{out_path}'{LogStyle.RESET}")
    return True


if __name__ == "__main__":
    print(f"\n{LogStyle.BOLD}--- ZERO-COLUMN FILTER SCRIPT ---{LogStyle.RESET}\n")

    if not OUTPUT_SUFFIX.strip():
        OUTPUT_SUFFIX = ".filtered"
        log_warn(f"Using default output suffix: \"{OUTPUT_SUFFIX}\"")

    # Filter out files that already have the output suffix to avoid double processing
    files_to_process = [f for f in INPUT_FILES if not f.endswith(OUTPUT_SUFFIX)]

    if not files_to_process:
        log_warn("No new CSV files found matching the criteria.")
    else:
        log_info(f"Found {len(files_to_process)} file(s) to process.\n")
        for in_path in files_to_process:
            base_name, ext = os.path.splitext(in_path)
            out_path = f"{base_name}{OUTPUT_SUFFIX}{ext}"
            _changed = process_csv(in_path, out_path)
            if _changed and OVERWRITE:
                log_warn(f"Overwriting file {in_path}")
                shutil.move(out_path, in_path)

    print(f"\n{LogStyle.BOLD}--- DONE ---{LogStyle.RESET}\n")
