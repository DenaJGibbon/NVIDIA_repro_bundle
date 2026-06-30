from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

data_root = Path("/data/ssl_v1/SSL_Clips")

out_csv = Path(
    "/home/nvidia/test_run/SSL_large_experiment/uploaded_filenames.csv"
)

out_csv.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Find audio files
# ------------------------------------------------------------

audio_exts = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg"
}

files = sorted([
    p for p in data_root.rglob("*")
    if p.is_file() and p.suffix.lower() in audio_exts
])

print("Found files:", len(files))


# ------------------------------------------------------------
# Build dataframe
# ------------------------------------------------------------

rows = []

for i, p in enumerate(files, start=1):

    if i % 10000 == 0:
        print(f"Processed {i} files")

    rows.append({
        "FilePath": str(p),
        "FileName": p.name,
        "Stem": p.stem,
        "Suffix": p.suffix.lower(),
        "ParentFolder": p.parent.name,
    })

df = pd.DataFrame(rows)

print(df.head())

df.to_csv(out_csv, index=False)

print("\nSaved:")
print(out_csv)