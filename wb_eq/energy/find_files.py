
## ----------------------------------------------------
## COMPACT VERSION
## ----------------------------------------------------
def find_files(dir_path, prefix, suffix, min_num=None, max_num=None, sort_natural=True, return_abs_path=False):
    import re; from pathlib import Path
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.is_dir(): raise FileNotFoundError(f"Directory '{dir_path}' not found.")
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    result_list = []
    for f in dir_path_obj.iterdir():
        if f.is_file():
            match = pattern.search(f.name)
            if match:
                num = int(match.group(1))
                if (min_num is None or num >= min_num) and (max_num is None or num <= max_num):
                    result_list.append(str(f.resolve()) if return_abs_path else str(f.relative_to(".")))

    if sort_natural:
        def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        result_list.sort(key=natural_sort_key)
    return result_list


if __name__ == '__main__':
    DCD_FILES = find_files("..", "amyl_wb_eq", ".dcd")  # TODO : trajectory dcd files
    print(DCD_FILES)


## ----------------------------------------------------
## FULL VERSION
## ----------------------------------------------------
# def find_files(dir_path, prefix, suffix, min_num=None, max_num=None, sort_natural=True, return_abs_path=False):
#     """
#     Finds files in a directory matching a specific prefix, numerical sequence, and suffix.
#
#     ex. find_files(".", "amyld_wb_eq", ".coor")
#     """
#     import re
#     from pathlib import Path
#
#     dir_path_obj = Path(dir_path)
#
#     # Uncomment the following lines to enable the directory check
#     # (matching your commented Tcl code)
#     if not dir_path_obj.is_dir():
#         raise FileNotFoundError(f"Directory '{dir_path}' not found.")
#
#     # Escape prefix and suffix so special characters (like '.') are treated literally
#     pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
#     result_list = []
#
#     # Iterate through all items in the directory
#     for f in dir_path_obj.iterdir():
#         if f.is_file():
#             match = pattern.search(f.name)
#             if match:
#                 # Extract the number and convert to integer (handles removing leading zeros)
#                 num = int(match.group(1))
#
#                 # Check boundaries (if min_num/max_num are provided)
#                 if (min_num is None or num >= min_num) and \
#                         (max_num is None or num <= max_num):
#
#                     if return_abs_path:
#                         # f.resolve() gets the absolute, normalized path
#                         fpath = str(f.resolve())
#                     else:
#                         # str(f) returns the path exactly as it was constructed (dir_path/filename)
#                         fpath = str(f.relative_to("."))
#
#                     result_list.append(fpath)
#
#     # Apply natural/dictionary sorting
#     if sort_natural:
#         def natural_sort_key(s):
#             # Splits the string into chunks of text and numbers, converting numbers to ints
#             # This mimics Tcl's `lsort -dictionary`
#             return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
#
#         result_list.sort(key=natural_sort_key)
#
#     return result_list
