import os
import pickle
import time
from typing import Counter, Tuple
from tqdm import tqdm

from data_loader.base import DataLoader
import torch
import numpy as np
import pandas as pd
from scipy.stats import skew

from data_pre_processing.base import DataPrePrecessing

PROTOCOLS = {
    'AVTP': 0,
    'PTP': 1,
    'IP_UDP': 2,
    'L2': 0,
}

class AEROPreProcessing(DataPrePrecessing):
    def initialize(self, data_loader: DataLoader) -> Tuple[np.ndarray, pd.DataFrame]:
        self.logger.info("Initializing pre processing...")
        start_time = time.time()
        
        path = self.get_output_path()

        if not os.path.exists(path + '/batch_0.pt'):
            data, target = data_loader.load()

            X, y = self.process(data, target)

            save_subset = self.config.get('pre_processing', {}).get('save_subset')
            if save_subset is not None:
                self.logger.warning(f">> Saving a subset of {save_subset}%")
                self.logger.warning(f">> Original shape: {X.shape}")
                starts = np.random.choice(range(len(X)), size=int(save_subset*len(X)), replace=False)
                self.logger.warning(f">> Final shape: {starts.shape}")
                y = y.iloc[starts].reset_index(drop=True)

            self.save((X, starts), y)
            del data, target, X, starts, y
        
        self.logger.info(f"Initializing finished in {time.time() - start_time}s")
        return self.load(path)

    def process(self, data: np.ndarray, target: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        window_stride   = self.config.get('pre_processing', {}).get('window_stride')
        window_size     = self.config.get('pre_processing', {}).get('window_size')

        X, y = self.__generate_features(data, target, window_stride=window_stride, window_size=window_size)
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
        protocol_filter   = self.config.get('data_loader', {}).get('protocol_filter')
        number_of_bytes = self.config.get('data_loader', {}).get('number_of_bytes') # TODO: THIS SHOULD NOT BE HERE
        save_subset     = self.config.get('pre_processing', {}).get('save_subset', None)

        file_name = (
            f"{phase}_"
            f"{method}_"
            f"wsize_{window_size}_"
            f"wstride_{window_stride}_"
            f"n_{number_of_bytes}_"
            f"{protocol_filter}"
        )
        
        if save_subset:
            file_name += f'_subset_{save_subset}'

        path = f'{processed_path}/{file_name}'

        os.makedirs(path, exist_ok=True)
        return path

    

    def _protocol_transition_matrix(self, protocols):
        N = 3
        T = np.zeros((N, N))

        for i in range(len(protocols) - 1):
            idx1 = PROTOCOLS[protocols[i]]
            idx2 = PROTOCOLS[protocols[i + 1]]
            T[idx1, idx2] += 1

        T = T / T.sum() if T.sum() > 0 else T
        return T
    
    def _timestamp_statistics(self, timestamps, protocols):
        N = 3

        stats = []
        for proto in range(N):
            proto_times = [timestamps[i] for i in range(len(protocols)) if PROTOCOLS[protocols[i]] == proto]
            if len(proto_times) < 2:
                stats.append(np.log10([1e7, 1e7, 1e7]))
                continue
            proto_times = [float(t) for t in proto_times]
            intervals = np.diff(proto_times)
            mean = np.mean(intervals)
            std = np.std(intervals) if np.std(intervals) > 0 else 1e-7
            skw = abs(skew(intervals)) if len(intervals) > 2 else 1e7
            stat = np.log10([mean, std, skw])
            stats.append(stat)
        return np.array(stats)
    
    def __generate_features(self, values: np.ndarray, labels: pd.DataFrame, window_size: int, 
                            window_stride: int) -> pd.DataFrame:
        self.logger.info("Generating features using AERO...")

        jump_payload_bytes      = 32
        number_payload_bytes    = 9

        start_payload = 14 + jump_payload_bytes # Assuming Ethernet header size
        final_payload = start_payload + number_payload_bytes

        X, y = list(), list()

        for i in tqdm(range(0, len(values) - window_size + 1, window_size)):
            # Compute a new (sliding window) index
            start_ix = i*(window_stride)
            end_ix = start_ix + window_size - 1 + 1

            # If index is larger than the size of the dataset, we stop
            if end_ix >= values.shape[0]:
                break

            window_values = values[start_ix:end_ix]
            window_labels = labels.iloc[start_ix:end_ix]

            # Assume labels DataFrame has: 'protocol', 'timestamp'
            protocols = window_labels['protocol'].tolist()
            timestamps = window_labels['timestamp'].tolist()
            payloads = window_values[:, start_payload:final_payload]

            T = self._protocol_transition_matrix(protocols)
            P = payloads
            S = self._timestamp_statistics(timestamps, protocols)

            feature_vector = np.concatenate([T.flatten(), P.flatten(), S.flatten()])

            # Labeling schema
            seq_y = self.__labeling_schema(window_labels)

            X.append(feature_vector)
            y.append(seq_y)

        x_array = np.array(X, dtype='float32')
        y_array = pd.DataFrame(y, columns=['label'])

        self.logger.info(f"Feature generation complete.")
        self.logger.info(f"Generated features shape: {x_array.shape}")
        self.logger.info(f"Generated labels: {y_array['label'].value_counts()}")

        return x_array, y_array

    def __labeling_schema(self, sequence: pd.DataFrame) -> bool:    
        seq_y = 'Normal'
        labels = self.get_label_mapping().values()
        labels = [label for label in labels if label != 'Normal']

        indexes = sequence['label'].value_counts().sort_values(ascending=False).reset_index()
        indexes_list = list(indexes['label'].values)

        set_attacks = set(labels)
        set_sequence_indexes = set(indexes_list)

        intersect = any(set_atk in set_sequence_indexes for set_atk in set_attacks)

        if intersect is True:
            attacks_mask = indexes['label'].isin(labels)
            indexes_attacks = indexes[attacks_mask]
            seq_y = indexes_attacks['label'].values[0]

        return seq_y

    def get_label_mapping(self):
        return {
            'Normal': 'Normal',
            'C_D': 'CAN DoS',
            'P_I': 'PTP Sync',
            'M_F': 'Switch MAC Flooding',
            'F_I': 'Frame Injection',
            'C_R': 'CAN Replay',
        }