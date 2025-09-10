import time
from typing import Tuple

import numpy as np
import pandas as pd
from scapy.all import rdpcap, raw

from data_loader.base import DataLoader


class TOWIDSLoader(DataLoader):
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load TOW-IDS dataset"""
        self.logger.info("Loading raw data...")
        start_time = time.time()

        raw_x_path      = self.config.get('data_loader', {}).get('raw_x_path')
        raw_y_path      = self.config.get('data_loader', {}).get('raw_y_path')
        number_of_bytes = self.config.get('data_loader', {}).get('number_of_bytes')

        raw_packets = rdpcap(raw_x_path)
        labels = pd.read_csv(raw_y_path, header=None, names=["index", "class", "label"])
        labels['label'] = labels['label'].map({
            'Normal': 'Normal',
            'C_D': 'CAN DoS',
            'P_I': 'PTP Sync',
            'M_F': 'Switch MAC Flooding',
            'F_I': 'Frame Injection',
            'C_R': 'CAN Replay',
        })
        
        n = len(raw_packets)
        
        # Preallocate arrays
        values = np.zeros((n, number_of_bytes), dtype=np.float32)
        timestamps = np.empty(n, dtype=object)   # float or object depending on pkt.time
        protocols = np.empty(n, dtype=object)    # string labels

        for i, pkt in enumerate(raw_packets):
            b = raw(pkt)
            m = min(len(b), number_of_bytes)
            if m:
                values[i, :m] = np.frombuffer(b, dtype=np.uint8, count=m)

            timestamps[i] = pkt.time
            protocols[i] = self.__detect_protocol_scapy(pkt)

        # Normalize in one vectorized step
        values /= 255.0

        # Assign to labels once, vectorized
        labels['timestamp'] = timestamps
        labels['protocol'] = protocols

        self.logger.info(f"Loading raw data finished in {(time.time() - start_time):.2f}s with shape of {values.shape}")
        return values, labels
    
    def __detect_protocol_scapy(self, pkt):
        """Detect protocol using Scapy's layer inspection."""
        if pkt.haslayer('UDP'):
            return 'UDP'
        if pkt.haslayer('802.1Q'):
            return 'AVTP'
        return 'OTHER'