import json
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Optional, Union

def record_trial(
    resistance: float,
    filepath: Union[str, Path],
    rise_times: Iterable[float],
    fall_times: Iterable[float],
    ionized_rise_times: Optional[List[Iterable[float]]] = None,
    ionized_fall_times: Optional[List[Iterable[float]]] = None,
):
    filepath = Path(filepath)
    entry = {
        "date": datetime.now().strftime("%-m-%-d"),
        "resistance": float(resistance),
        "rise_times": list(rise_times),
        "fall_times": list(fall_times),
        "ionized_rise_times": [list(x) for x in (ionized_rise_times or [])],
        "ionized_fall_times": [list(x) for x in (ionized_fall_times or [])],
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def load_data(filepath, data):
    filepath = Path(filepath)
    if not filepath.exists():
        return
    
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            data.add_trial_from_data(
                entry["rise_times"],
                entry["fall_times"],
                entry["date"],
                entry["resistance"],
                entry.get("ionized_rise_times", []),
                entry.get("ionized_fall_times", [])
            )

def remove_trial_by_index(filepath, index):
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        trials = [json.loads(line) for line in f if line.strip()]

    if index < 0 or index >= len(trials):
        raise IndexError("Trial index out of range")

    del trials[index]

    with filepath.open("w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t) + "\n")

def remove_trial_by_match(filepath, *, rise_times, fall_times):
    filepath = Path(filepath)

    with filepath.open("r", encoding="utf-8") as f:
        trials = [json.loads(line) for line in f if line.strip()]

    filtered = [
        t for t in trials
        if not (t["rise_times"] == rise_times and t["fall_times"] == fall_times)
    ]

    if len(filtered) == len(trials):
        raise ValueError("No matching trial found")

    with filepath.open("w", encoding="utf-8") as f:
        for t in filtered:
            f.write(json.dumps(t) + "\n")