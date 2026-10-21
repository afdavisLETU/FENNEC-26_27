#Imports
import os 
import glob
import torch
import pandas as pd
import random
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

#excel_dir = data_dir = r"C:\Users\Mitch\OneDrive - LeTourneau University\Andrew Davis's files - Project FENNEC\FENNEC 2026-2027\Data Science Team\Mitchell Moffitt\Python Code\Jupyter Notebooks\GOAL 3- Data preprocessing\Raw_data"
#csv_dir = r"C:\Users\Mitch\OneDrive - LeTourneau University\Andrew Davis's files - Project FENNEC\FENNEC 2026-2027\Data Science Team\Mitchell Moffitt\Python Code\Jupyter Notebooks\GOAL 3- Data preprocessing\Processed_data"
CG_STEP = 0.15   # CG positions are multiples of this (inches)
CG_MAX = 0.75    # largest CG offset in either direction (inches)
#The Wind levels are not accounted for but shouldnt matter too much

#===============================================================
def extract_cg_code_Var(filepath):
    #This is were the file is parsed to extract the CG code
    filename = os.path.basename(filepath)
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    if len(parts) < 2:
        raise ValueError(f"Invalid file format: {filename}")

    cg_code = parts[1]

    # Format: a decimal magnitude + a direction letter (F, M, or A)
    # e.g. "0.15A" (aft), "0.15F" (forward), "0.00M" (middle/neutral)
    if not re.match(r'^\d+(\.\d+)?[FMA]$', cg_code):
        raise ValueError(f"Invalid CG code format '{cg_code}' in {filename}")

    return cg_code

#===============================================================
def extract_cg_label(filepath):
    # Pull the CG code from the filename (e.g. "0.15A", "0.00M", "0.45F").
    # extract_cg_code already confirmed it matches: digits + optional decimal + F/M/A
    code = extract_cg_code_Var(filepath)

    # Split the code into its two parts:
    #   "0.15A" -> magnitude_str = "0.15", direction_char = "A"
    magnitude_str, direction_char = code[:-1], code[-1]
    magnitude = round(float(magnitude_str), 2)   # round() avoids float noise

    # Direction map: letter -> sign applied to the magnitude.
    # Forward is positive, aft is negative, middle is zero.
    # (Flip F and A here if you'd rather have aft positive.)
    direction_map = {'F': 1, 'M': 0, 'A': -1}

    # The only magnitudes that exist in your test matrix:
    # 0.15 in steps, up to 0.75 in
    valid_magnitudes = {0.00, 0.15, 0.30, 0.45, 0.60, 0.75}

    # Validate before using, so a bad filename fails loudly
    # and tells you exactly which file caused the problem.
    if direction_char not in direction_map:
        raise ValueError(f"Invalid direction '{direction_char}' in {filepath}")
    if magnitude not in valid_magnitudes:
        raise ValueError(f"Invalid CG magnitude {magnitude} in {filepath}")
    if direction_char == 'M' and magnitude != 0.00:
        raise ValueError(f"Middle ('M') must be 0.00, got {magnitude} in {filepath}")
    if direction_char != 'M' and magnitude == 0.00:
        raise ValueError(f"'{direction_char}' can't have magnitude 0.00 in {filepath}")

    # Combine into a single signed number (the label for your model)
    return direction_map[direction_char] * magnitude

#===============================================================
def get_cg_offsets(data_dir, ext="xlsx"):
    """
    Returns a list of CG offsets (floats, in inches), one per file
    in data_dir, using extract_cg_label for parsing/validation.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = sorted(glob.glob(os.path.join(data_dir, f"*.{ext}")))
    if not files:
        raise ValueError(f"No .{ext} files found in {data_dir}")

    offsets = []
    for file in files:
        offsets.append(extract_cg_label(file))  # raises ValueError on bad filenames

    return offsets

#===============================================================
def scale_target(y):
    """Offsets in [-0.75, 0.75] -> [-1, 1]. Needs no fitting."""
    return np.asarray(y, dtype=np.float32) / CG_MAX
 
#===============================================================
def unscale_target(y_scaled):
    """Model output -> inches. Use this before computing MAE/RMSE."""
    return np.asarray(y_scaled) * CG_MAX

#===============================================================
class Flight_Var:
    def __init__(self, filepath, feature_cols=None):
        self.filepath = filepath #full path to CSV 
        self.filename = os.path.basename(filepath)
        self.cg_code = extract_cg_code_Var(filepath)        

        # Not sure waht they are for yet
        self.cg_label = extract_cg_label(filepath)
        self.data = self.load_data_var()                   # (timesteps, features)
        self._scaled = False
        self.feature_cols = feature_cols
 
    def load_data_var(self):
        # load CSV,return as np.array - load vars of intrest?
        df = pd.read_csv(self.filepath)
        return df.values

#===============================================================
class Flight_Objects_Var:
    def __init__(self, csv_dir):
        self.csv_files = sorted(glob.glob(os.path.join(csv_dir,"*.csv")))
        self.flights = [Flight_Var(f) for f in self.csv_files]

        #Not sure about this yet
    def get_data_and_labels_Var(self):
        # Flights have different lengths, so X stays a list (np.array would fail on ragged arrays).
        # Labels are the signed offsets in inches (cg_label), not the "0.15A" strings.
        X = [f.data for f in self.flights]
        y = np.array([f.cg_label for f in self.flights], dtype=np.float32)
        return X, y
    
#===============================================================
#EXTRA CODE: THIS IS FOR SLITTING THE DATA MANUALLY RATHER THAN A PERCENTAGE
#===============================================================

# def split_flights_by_class(flight_objects, train_per_class=3, val_per_class=0, seed=42):
#     random.seed(seed)
#     flights_by_class = {}

#     for flight in flight_objects.flights:
#         key = flight.cg_code
#         flights_by_class.setdefault(key, []).append(flight)

#     train_flights = []
#     val_flights = []
#     test_flights = []

#     for key, flights in flights_by_class.items():
#         if len(flights) < train_per_class + val_per_class + 1:
#             raise ValueError(f"Not enough flights for label {key}")

#         shuffled = flights.copy()
#         random.shuffle(shuffled)

#         train_flights.extend(shuffled[:train_per_class])
#         val_flights.extend(shuffled[train_per_class:train_per_class + val_per_class])
#         test_flights.extend(shuffled[train_per_class + val_per_class:train_per_class + val_per_class + 1])

#     return train_flights, val_flights, test_flights

#===============================================================
#SPLITS DATA BY FRACTION OF FLIGHTS PER CG POSITION
#===============================================================
def split_flights_by_fraction(flight_objects, train_frac=0.8, val_frac=0.0, seed=42):
    """
    Split flights per CG position by percentage. Whatever is left after
    train and val goes to test. Every CG position appears in every split.
    """
    rng = random.Random(seed)
    by_cg = {}
    for f in flight_objects.flights:
        by_cg.setdefault(f.cg_code, []).append(f)

    train, val, test = [], [], []
    for cg, flights in by_cg.items():
        n = len(flights)
        n_val = int(n * val_frac + 0.5)
        n_train = int(n * train_frac + 0.5)
        n_train = min(n_train, n - n_val - 1)       # always keep at least 1 test flight
        if n_train < 1:
            raise ValueError(f"CG {cg}: only {n} flights, not enough to split")

        shuffled = flights.copy()
        rng.shuffle(shuffled)
        train += shuffled[:n_train]
        val += shuffled[n_train:n_train + n_val]
        test += shuffled[n_train + n_val:]
    return train, val, test

#===============================================================
class FlightDataNormalizer_Var:
    def __init__(self):
        self.scaler = None

    def fit(self, flights):
        # Fit the scaler to the train flight objects

        # Concatenate all flight data along axis 0 (timesteps)
        all_train_data = np.vstack([f.data for f in flights])
        self.scaler = StandardScaler()
        self.scaler.fit(all_train_data)
        print(f"Scaler fitted: mean shape {self.scaler.mean_.shape} var shape {self.scaler.var_.shape}")

        # Print mean and std for each feature
        print("Scaler fitted on training data:")
        for i, (mean, std) in enumerate(zip(self.scaler.mean_, np.sqrt(self.scaler.var_))):
            print(f"Feature {i}: mean = {mean:.4f}, std = {std:.4f}")

    def transform(self,flights):
        # Apply scaler to val/test data

        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted yet")
        
        # transform() changes f.data in place, so re-running a cell would scale
        # already-scaled data a second time. The _scaled flag stops that.
        for f in flights:
            if f._scaled:
                raise RuntimeError(f"{f.filename} is already normalized; reload the flights first")
            f.data = self.scaler.transform(f.data)
            f._scaled = True
        return flights
    
    def fit_transform(self, train_flights):
        self.fit(train_flights)
        return self.transform(train_flights)

#===============================================================
class FlightDataSegmenter_Var:
    def __init__(self, segment_length, stride=None, max_segments_per_flight=None, seed=0):
        self.segment_length = segment_length
        self.stride = stride or segment_length      # stride < length -> overlap
        self.max_segments_per_flight = max_segments_per_flight
        self.rng = np.random.default_rng(seed)

    def segment_flights(self, flights):
        segments, offsets, flight_idx = [], [], []
        L = self.segment_length
        shortest = min((f.data.shape[0] for f in flights), default=L)
        if shortest < L:
            print(f"Warning: shortest flight has {shortest} rows but segment_length is {L}")

        for i, f in enumerate(flights):
            n_timesteps = f.data.shape[0]
            starts = np.arange(0, n_timesteps - L + 1, self.stride)
            
            if len(starts) == 0:
                print(f"Skipping {f.filename}: shorter than segment_length")
                continue
            
            if (self.max_segments_per_flight is not None
                    and len(starts) > self.max_segments_per_flight):
                starts = self.rng.choice(starts, self.max_segments_per_flight, replace=False)

            for s in starts:
                segments.append(f.data[s:s + L])
                offsets.append(f.cg_label)
                flight_idx.append(i)

        return (
            np.stack(segments).astype(np.float32),            # (N, L, F)
            scale_target(offsets),                            # (N,)  in [-1, 1]
            np.array(flight_idx),                             # (N,)  for per-flight averaging
        )
    
#===============================================================
def make_dataloader_Var(segments, targets, batch_size=32, shuffle=True):
    X = torch.tensor(segments, dtype=torch.float32)
    #Targets it just used as a variable that shows what the model is predicting
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)   # (N, 1) for MSELoss

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    return loader
