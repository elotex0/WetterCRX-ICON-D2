import os
import sys
import json
from datetime import datetime

png_root = sys.argv[1]  # z.B. icond2/<RUN>
run = sys.argv[2]
date = sys.argv[3] if len(sys.argv) > 3 else datetime.utcnow().strftime("%Y%m%d")

metadata = {
    "run": run,
    "date": date,
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "var_types": {},
}

for var_type in os.listdir(png_root):
    var_path = os.path.join(png_root, var_type)
    if not os.path.isdir(var_path):
        continue
    files = sorted(f for f in os.listdir(var_path) if f.endswith(".png"))
    timesteps = [f.split("_")[-1].replace(".png","") for f in files]
    metadata["var_types"][var_type] = {
        "num_steps": len(files),
        "timesteps": timesteps
    }

# metadata.json liegt **außerhalb** des run-Ordners
meta_path = os.path.join(os.path.dirname(png_root), "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata written to {meta_path}")
