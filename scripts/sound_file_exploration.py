from glob import glob
import pandas as pd
from pathlib import Path

files = sorted(glob("/home/nvidia/data/ssl_v1/SSL_Clips_small/*.wav"))

df = pd.DataFrame({
    "OutputPath": files,
    "ClipID": [Path(f).stem for f in files]
})

df.to_csv(
    "/home/nvidia/data/ssl_v1/SSL_Clips_small/ssl_smoke_manifest_brev.csv",
    index=False
)

print(df.head())
print(len(df))
