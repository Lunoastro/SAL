import fnmatch
import os


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def read_audio(path, tries=5, wait=10):
    import time
    import soundfile as sf
    for i in range(tries):
        try:
            return sf.read(path)
        except sf.LibsndfileError:
            if i == tries - 1:
                raise
            print(f"read failed ({i + 1}/{tries}), retrying in {wait}s: {path}", flush=True)
            time.sleep(wait)
