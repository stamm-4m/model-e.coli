import yaml
from pathlib import Path
import os

def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def save_yaml(data, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_br_id(dataset):
    """
    Extract BR02, BR03, ... from dataset filename.
    """

    name = os.path.basename(dataset.path)
    return name.replace(".xls", "")
