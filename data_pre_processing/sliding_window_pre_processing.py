import os
import pickle
import time
from typing import Counter, Tuple

from data_loader.base import DataLoader
import torch
import numpy as np
import pandas as pd

from data_pre_processing.base import DataPrePrecessing


class SlidingWindowPrePrecessing(DataPrePrecessing):
    def initialize(self, data_loader: DataLoader) -> Tuple[np.ndarray, pd.DataFrame]:
        self.logger.info("Initializing pre processing...")
        start_time = time.time()
        
        path = self.get_output_path()

        if not os.path.exists(path + 'batch_0.pt'):
            data, target = data_loader.load()

            (X_view, starts), y = self.process(data, target)

            save_subset = self.config.get('pre_processing', {}).get('save_subset')
            if save_subset is not None:
                self.logger.warning(f">> Saving a subset of {save_subset}%")
                indices = np.random.choice(len(starts), size=int(save_subset*len(starts)), replace=False)
                starts = starts[indices]
                y = y.iloc[indices].reset_index(drop=True)

            self.save((X_view, starts), y)
            del data, target, indices, X_view, starts, y
        
        self.logger.info(f"Initializing finished in {time.time() - start_time}s")
        return self.load(path)


    def process(self, data: np.ndarray, target: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        window_stride   = self.config.get('pre_processing', {}).get('window_stride')
        window_size     = self.config.get('pre_processing', {}).get('window_size')
        max_windows     = self.config.get('pre_processing', {}).get('max_windows', None)

        X, y = self.__get_window_data(data, target, window_stride=window_stride, window_size=window_size, 
                                         max_windows=max_windows)
        return X, y


    def save(self, X: np.ndarray, y: pd.DataFrame):
        self.logger.info("Saving data in cash file in batchs...")

        batch_size = self.config.get('pre_processing', {}).get('save_batch_size', len(X))
        path = self.get_output_path()

        X_view, starts = X

        # Split the data into manageable batches
        num_batches = len(starts) // batch_size + 1

        for i in range(num_batches):
            batch_starts = starts[i*batch_size : (i+1)*batch_size]
            batch_X = X_view[batch_starts]
            batch_y = y.iloc[i*batch_size : (i+1)*batch_size]

            # Save each batch with an index to maintain order
            batch_path = f"{path}/batch_{i}.pt"
            torch.save({'X': batch_X, 'y': batch_y}, batch_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Batch {i+1}/{num_batches} saved to {batch_path}")
    
    
    def load(self, path: str) -> tuple[np.ndarray, pd.DataFrame]:
        self.logger.info(f"Loading cached data from: {path}")
        
        files = [f for f in os.listdir(path)]
        files.sort()
        all_X, all_y = [], []

        for file in files:
            cache = torch.load(f"{path}/{file}", weights_only=False)
            all_X.append(cache['X'])
            all_y.append(cache['y'])

        X = np.concatenate(all_X, axis=0)
        y = pd.concat(all_y, axis=0).reset_index(drop=True)

        load_subset = self.config.get('pre_processing', {}).get('load_subset')
        if load_subset is not None:
            self.logger.warning(f"Loading data with subset of {load_subset}%")

            indices = np.random.choice(len(X), size=int(load_subset*len(X)), replace=False)
            X = X[indices]
            y = y.iloc[indices].reset_index(drop=True)

        return X, y
    
    
    def get_output_path(self) -> str:
        """
        Get the output path for the processed data.
        """
        phase           = self.config.get('phase')
        processed_path  = self.config.get('pre_processing', {}).get('processed_path')
        method          = self.config.get('pre_processing', {}).get('name')
        window_size     = self.config.get('pre_processing', {}).get('window_size')
        window_stride   = self.config.get('pre_processing', {}).get('window_stride')
        save_subset     = self.config.get('pre_processing', {}).get('save_subset', None)

        file_name = (
            f"{phase}_"
            f"{method}_"
            f"wsize_{window_size}_"
            f"wstride_{window_stride}_"
        )
        
        if save_subset:
            file_name += f'_subset_{save_subset}'

        path = f'{processed_path}/{file_name}'

        os.makedirs(path, exist_ok=True)
        return path
    

    def __labeling_schema_vectorized(self, desc_windows: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
        """
        Vectorized labeling:
        - Input: desc_windows: ndarray of shape (num_windows, window_size), dtype object/str
        - Output: ndarray of shape (num_windows,), dtype str
        - Rule: pick the most frequent non-'Normal' label in each row; if none, 'Normal'.
        - Ties: break by lexicographic order of the label string.
        """
        if desc_windows.size == 0:
            return np.array([], dtype=object)

        atk = [i for i in labels['label'].unique() if i != 'Normal']
        normal = "Normal"

        # Factorize all labels at once to integer codes
        flat = desc_windows.ravel()
        codes, uniques = pd.factorize(flat)              # codes in [0..K-1], uniques: array of labels
        codes = codes.reshape(desc_windows.shape)

        # Mask invalid (e.g., all-NaN rows) — pd.factorize uses -1 for NaN/None
        # We’ll just ignore -1 when counting.
        num_labels = len(uniques)
        if num_labels == 0:
            return np.full(desc_windows.shape[0], normal, dtype=object)

        # Count occurrences of each label per row (vectorized over rows)
        # Using one-hot accumulation without big temporary allocations.
        counts = np.zeros((desc_windows.shape[0], num_labels), dtype=np.int32)
        # Add counts column by column (window_size is small, so this is fast and memory-light)
        n_rows = desc_windows.shape[0]
        row_idx = np.arange(n_rows)
        for j in range(desc_windows.shape[1]):
            col = codes[:, j]
            valid = col >= 0
            np.add.at(counts, (row_idx[valid], col[valid]), 1)

        # Identify which global labels are "attack" labels
        uniques_str = uniques.astype(str)
        attack_mask = np.isin(uniques_str, list(atk))
        if not attack_mask.any():
            # No attack labels present anywhere
            return np.full(n_rows, normal, dtype=object)

        # Restrict to attack labels
        counts_attack = counts[:, attack_mask]                      # (n_rows, n_attack_labels)
        attack_labels = uniques_str[attack_mask]                    # (n_attack_labels,)

        # Row-wise max count among attacks
        max_attack = counts_attack.max(axis=1)                      # (n_rows,)
        has_attack = max_attack > 0

        # Tie-break lexicographically among attack labels
        # Precompute a rank (0 = smallest lexicographic label)
        lex_ranks = np.argsort(attack_labels).argsort()             # rank per attack label index
        # Among positions with count == max_attack, choose smallest lex rank.
        tie_mask = counts_attack == max_attack[:, None]             # (n_rows, n_attack_labels)
        # Use -lex_ranks so that argmax picks the smallest rank; set non-ties to -inf
        prefs = np.where(tie_mask, -lex_ranks, -np.inf)
        best_j = np.argmax(prefs, axis=1)                           # (n_rows,)

        # Build the result
        out = np.full(n_rows, normal, dtype=object)
        out[has_attack] = attack_labels[best_j[has_attack]]
        return out


    def __get_window_data(self, data: np.ndarray, target: pd.DataFrame, window_stride: int=1, window_size: int=2,
                        max_windows: int=None, rng=None):
        rng = np.random.default_rng(rng)
        n = data.shape[0]
        last_start = n - window_size
        if last_start < 0:
            return [], []

        # all candidate starts given the stride
        starts = np.arange(0, last_start + 1, window_stride)

        # --- lightweight label-side windows (cheap) ---
        desc = target['label'].to_numpy()
        ts = target['timestamp'].to_numpy()
        desc_windows_all = np.lib.stride_tricks.sliding_window_view(desc, window_shape=window_size)

        # subsample starts *before* touching big value windows
        if max_windows is not None and len(starts) > max_windows:
            starts = rng.choice(starts, size=max_windows, replace=False)
            starts.sort()  # keep roughly chronological order

        # compute labels only for chosen starts
        seq_y = self.__labeling_schema_vectorized(desc_windows_all[starts], target)
        start_times = ts[starts]
        desc_windows = desc_windows_all[starts]

        # --- now build value windows for the chosen starts only ---
        # sliding_window_view returns a view; avoid materializing everything
        X_view = np.lib.stride_tricks.sliding_window_view(data, window_shape=(window_size,) + data.shape[1:])
        X_view = X_view.reshape(n - window_size + 1, window_size, *data.shape[1:])
        # X = X_view[starts]

        self.logger.info("Class distribution:")
        self.logger.info(Counter(seq_y))

        y = list(zip(seq_y, start_times.astype(float), desc_windows))
        y = pd.DataFrame(y, columns=['label', 'start_time', 'desc_windows'])
        return (X_view, starts), y
    