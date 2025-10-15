import time
from typing import Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
from scapy.all import raw, PcapReader, Ether

from data_loader.base import DataLoader


class TOWIDSLoader(DataLoader):
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load TOW-IDS dataset"""
        self.logger.info("Loading raw data...")
        start_time = time.time()

        raw_x_path      = self.config.get('data_loader', {}).get('raw_x_path')
        raw_y_path      = self.config.get('data_loader', {}).get('raw_y_path')
        number_of_bytes = self.config.get('data_loader', {}).get('number_of_bytes')
        protocol_filter = self.config.get('data_loader', {}).get('protocol_filter')

        labels = pd.read_csv(raw_y_path, header=None, names=["pkt_idx", "class", "label"])
        labels['label'] = labels['label'].map({
            'Normal': 'Normal',
            'C_D': 'CAN DoS',
            'P_I': 'PTP Sync',
            'M_F': 'Switch MAC Flooding',
            'F_I': 'Frame Injection',
            'C_R': 'CAN Replay',
        })

        n = len(labels)
        values      = np.empty((n, number_of_bytes), dtype=np.float32)
        timestamps  = np.empty(n, dtype=object)
        protocols   = np.empty(n, dtype=object)

        with PcapReader(raw_x_path) as pcap_reader:
            for i, pkt in tqdm(enumerate(pcap_reader), total=n):
                protocol = self.__detect_protocol_scapy(pkt)
                if (protocol_filter and (
                    (type(protocol_filter) == list and protocol not in protocol_filter) or 
                    (type(protocol_filter) != list and protocol != protocol_filter)
                    )):
                    continue

                b = raw(pkt)
                m = min(len(b), number_of_bytes)
                arr = np.frombuffer(b, dtype=np.uint8, count=m)
                if len(b) < number_of_bytes:
                    arr = np.pad(arr, (0, number_of_bytes - len(b)), 'constant')
                
                protocols[i]    = protocol
                timestamps[i]   = pkt.time
                values[i]       = arr

        values /= 255.0

        labels['timestamp'] = timestamps
        labels['protocol'] = protocols
        labels.dropna(inplace=True)

        valid_idx = labels.index
        values = values[valid_idx]

        self.logger.info(f"Loading raw data finished in {(time.time() - start_time):.2f}s with shape of {values.shape}")
        return values, labels
    
    def __detect_protocol_scapy(self, pkt):
        """Detect protocol using Scapy's layer inspection."""
        eth_type = pkt[Ether].type

        if eth_type in [2054, 35061, 8938]:
            return 'L2'

        if eth_type in [2048, 34525]:
            return 'IP_UDP'
        
        if eth_type in [33024, 8944]:
            return 'AVTP'
        
        if eth_type in [35063]:
            return 'PTP'
        
        return '-1'