import os


def norm_str(s: str) -> str:
    return (
        str(s)
        .replace("\ufeff", "")
        .replace('"', "")
        .replace("'", "")
        .strip()
        .lower()
    )


def alt_names(filename: str) -> list:
    """
    Build normalized filename variants for robust matching.
    """
    fn = norm_str(filename)
    base = norm_str(os.path.basename(fn))
    out = {fn, base}
    if base.endswith(".csv"):
        out.add(base[:-4])
    # Handle common FP/F aliases used in filenames
    if base.startswith("fp"):
        out.add("f" + base[2:])
    if base.startswith("f") and not base.startswith("fp"):
        out.add("fp" + base[1:])
    return [x for x in out if x]
