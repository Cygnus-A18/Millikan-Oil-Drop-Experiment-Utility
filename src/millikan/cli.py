import time
import json
import argparse
from pathlib import Path
from .terminal import get_key
from .io import record_trial, load_data
from .analysis import compute_e_from_all_points, refine_e, fit_e_multistart, compute_e_from_lowest_points
from .models import DropletData
from .plotting import plot_discrete_charge, plot_each_ionization, generate_table_latex, generate_table_plaintext

def record_trial_live(filepath):
    """
    Interactive terminal stopwatch recorder.

    Controls:
      - Press Enter to record current elapsed time (alternates fall/rise)
      - Type 'q' + Enter to finish and save
      - Type 'i' to start recording a new ionization
      - Type 'p' to pause recording and reset timer
      - Type 'r' while paused to resume recording
    """

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fall_times = []
    rise_times = []

    ionized_fall_times = []
    ionized_rise_times = []

    current_fall = fall_times
    current_rise = rise_times

    history = []

    next_kind = "fall"
    running = True
    t0 = time.perf_counter()

    while True:
        print("press enter to begin recording.")
        cmd = get_key().lower()

        if cmd in ("\r", "\n", " "):
            t0=time.perf_counter()
            break



    print("Stopwatch running.")
    print("Enter = record | q = finish & save | i = new ionization")

    while True:
        cmd = get_key().lower()

        if (cmd == "q"):
            break
        elif (cmd == "p"):
            running = False
            t0 = None
            print("Paused.")
            continue
        elif (cmd == 'r'):
            if running:
                continue
            else:
                running = True
                t0 = time.perf_counter()
                print("Resumed.")
            continue

        if not running:
            print("Paused. Press 'r' to resume.")
            continue

        elif cmd == "i":
            ionized_fall_times.append([])
            ionized_rise_times.append([])
            current_fall = ionized_fall_times[-1]
            current_rise = ionized_rise_times[-1]
            print(f"Started ionized segment #{len(ionized_fall_times)}")
            continue
        
        elif cmd in ("\r", "\n", " "):
            elapsed = time.perf_counter() - t0
            t0 = time.perf_counter()

            if next_kind == "fall":
                current_fall.append(elapsed)
                history.append("fall")
                next_kind = "rise"
                print(f"Recorded fall: {elapsed:.3f}s")
            else:
                current_rise.append(elapsed)
                history.append("rise")
                next_kind = "fall"
                print(f"Recorded rise: {elapsed:.3f}s")

    entry = {
        "rise_times": rise_times,
        "fall_times": fall_times,
        "ionized_rise_times": ionized_rise_times,
        "ionized_fall_times": ionized_fall_times,
    }

    print("\n--- Recorded trial ---")
    print(f"Neutral: {len(fall_times)} fall, {len(rise_times)} rise")
    print(f"Ionized segments: {len(ionized_fall_times)}")
    for idx, (ff, rr) in enumerate(zip(ionized_fall_times, ionized_rise_times), start=1):
        print(f"  Segment {idx}: {len(ff)} fall, {len(rr)} rise")
    print("\nFull entry:")
    print(json.dumps(entry, indent=2))
    print("----------------------\n")

    while True:
        confirm = input("Save this trial? (y/n) or x for manual override: ").strip().lower()
        if confirm in ("y", "n", "x"):
            break
        print("Please type 'y' or 'n'.")

    if confirm == "y":
        R = float(input("Enter Resistance of Trial: ").strip().lower())
        record_trial(
            resistance=R,
            filepath=filepath,
            rise_times=rise_times,
            fall_times=fall_times,
            ionized_rise_times=ionized_rise_times,
            ionized_fall_times=ionized_fall_times,
        )
        print(f"Saved to {filepath}")
    elif confirm == "n":
        print("Discarded (not saved).")
    else:
        breakpoint()

def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--record",
        metavar="FILENAME",
        help="Record a live trial to the given file",
    )

    group.add_argument(
        "--open",
        dest="open_file",
        metavar="FILENAME",
        help="Open a recorded trial file",
    )

    group.add_argument(
        "--evaluate_e",
        metavar="FILENAME",
        help="Evaluate elementary charge from a recorded trial file",
    )

    args = parser.parse_args()

    if args.record is not None:
        record_trial_live(args.record)
    elif args.open_file is not None:
        data = DropletData()
        load_data(args.open_file, data)
        plot_discrete_charge(data)
        plot_discrete_charge(data, max_size=10.0)
        plot_discrete_charge(data, show_ionization_measurements=False, show_mean_q_lines=False, max_size=10.0)
        plot_discrete_charge(data, show_ionization_measurements=False, show_mean_q_lines=False)
        plot_discrete_charge(data, show_ionization_measurements=False)
        plot_discrete_charge(data, show_ionization_measurements=False, max_size=10.0)
        plot_discrete_charge(data, show_mean_q_lines=False)
        plot_discrete_charge(data, show_mean_q_lines=False, max_size=10.0)
        plot_discrete_charge(data, show_only_points=True)
        plot_each_ionization(data)
        generate_table_latex(data, "data/data_table_tex.txt")
        generate_table_plaintext(data, "data/data_table.txt")
    elif args.evaluate_e is not None:
        data = DropletData()
        load_data(args.evaluate_e, data)
        e1 = compute_e_from_lowest_points(data)
        e2 = compute_e_from_all_points(data)
        e3, _ = fit_e_multistart(data)

        print(f'lowest points: {e1}\nall points: {e2}\nfit e value: {e3}')
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
