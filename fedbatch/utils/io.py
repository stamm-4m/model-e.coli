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


def get_time_ranges(yaml_dict: dict, br_id: str):
    try:
        time_sb = yaml_dict["bioreactor"][br_id]["t_sb"]["value"]
        time_ind = yaml_dict["bioreactor"][br_id]["t_ind"]["value"]
    except KeyError as e:
        raise KeyError(f"Missing time parameters for {br_id}: {e}")

    if time_sb >= time_ind:
        raise ValueError(
            f"Invalid time ranges for {br_id}: "
            f"t_sb ({time_sb}) >= t_ind ({time_ind})"
        )

    return time_sb, time_ind

def get_br_id(dataset):
    """
    Extract BR02, BR03, ... from dataset filename.
    """

    name = os.path.basename(dataset.path)
    return name.replace(".xls", "")
