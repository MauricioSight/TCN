import os
import pickle
import time
from typing import Tuple
import torch
from tqdm import tqdm

import numpy as np
import pandas as pd
from scapy.all import raw, PcapReader, Ether

from data_loader.base import DataLoader


class AEIDLoader(DataLoader):
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Load AEID dataset"""
        
        phase           = self.config.get('phase')
        processed_path  = self.config.get('pre_processing', {}).get('processed_path')
        raw_x_path      = self.config.get('data_loader', {}).get('raw_x_path')
        injected_path   = self.config.get('data_loader', {}).get('injected_path')
        number_of_bytes = self.config.get('data_loader', {}).get('number_of_bytes')
        protocol_filter = self.config.get('data_loader', {}).get('protocol_filter')
        cash_path       = processed_path + f"/{phase}_AEID_{number_of_bytes}.pt"

        if os.path.exists(cash_path):
            self.logger.info("Loaded data from cash")
            cache = torch.load(cash_path, weights_only=False)
            return cache['values'], cache['labels']

        self.logger.info("Loading raw data...")
        start_time = time.time()

        # Read injected path
        injected_stream = []
        with PcapReader(injected_path) as pcap_reader:
            for i, pkt in tqdm(enumerate(pcap_reader)):
                b = raw(pkt)
                m = min(len(b), number_of_bytes)
                arr = np.frombuffer(b, dtype=np.uint8, count=m)
                if len(b) < number_of_bytes:
                    arr = np.pad(arr, (0, number_of_bytes - len(b)), 'constant')
                
                injected_stream.append(arr)

        values      = []
        timestamps  = []
        protocols   = []
        with PcapReader(raw_x_path) as pcap_reader:
            for i, pkt in tqdm(enumerate(pcap_reader)):
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
                
                protocols.append(protocol)
                values.append(arr)
                timestamps.append(pkt.time)

        injected_stream = np.array(injected_stream)
        values          = np.array(values)

        labels = self.__generate_labels(values, injected_stream)
        labels = pd.DataFrame(labels, columns=['label'])
        labels['pkt_idx'] = timestamps
        labels['timestamp'] = timestamps
        labels['protocol'] = protocols

        values = np.float32(values)
        values /= 255.0

        torch.save({'values': values, 'labels': labels}, cash_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Generated labels: {labels['label'].value_counts()}")

        self.logger.info(f"Loading raw data finished in {(time.time() - start_time):.2f}s with shape of {values.shape}")
        return values, labels
    
    def __generate_labels(self, packets_list, injected_packets):
        labels_list = []

        for packet in packets_list:
            current_label = 'Normal'

            if self.__is_array_in_list_of_arrays(packet, injected_packets):
                current_label = 'Replay'

            labels_list.append(current_label)

        return labels_list
    
    def __is_array_in_list_of_arrays(self, array_to_check, list_np_arrays):
        # Reference:
        # https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        is_in_list = np.any(np.all(array_to_check == list_np_arrays, axis=1))

        return is_in_list    
    
    def __detect_protocol_scapy(self, pkt):
        """Detect protocol using Scapy's layer inspection."""
        eth_type = pkt[Ether].type

        if eth_type in [2054, 35061, 8938]:
            return 'L2'

        if eth_type in [2048, 34525]:
            return 'IP_UDP'
        
        if pkt.haslayer('802.1Q'):
            return 'AVTP'
        
        if eth_type in [35063]:
            return 'PTP'
        
        return '-1'