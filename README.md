# Millikan-Oil-Drop-Experiment-Utility
Helper utility to serialize droplet data and perform data analysis and produce figures from custom data format. Includes a custom terminal based stopwatch to collect live data more efficiently.

# Usage:

build: pip install -e .

run options: 
- millikan --open "filename": opens specified file plots all points and ionizations and then breakpoints for manual edits.
- millikan --record "filename": starts running stop watch app and saves recorded data to the specified file.


# Stopwatch Usage:

Controls:
- Press Enter to start recording
- Press Enter to record current elapsed time (alternates fall/rise)
- Type 'q' + Enter to finish and save
- Type 'i' to start recording a new ionization
- Type 'p' to pause recording and reset timer
- Type 'r' while paused to resume recording
