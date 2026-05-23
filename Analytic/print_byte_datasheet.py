from scapy.all import raw, PcapReader, Ether

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

import numpy as np
import pandas as pd
from tqdm import tqdm
import pyshark

raw_x_path      = "./data/tow-ids-dataset/raw/Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap"
raw_y_path      = "./data/tow-ids-dataset/raw/y_train.csv"
number_of_bytes = 58

labels = pd.read_csv(raw_y_path, header=None, names=["index", "class", "label"])
labels['label'] = labels['label'].map({
    'Normal': 'Normal',
    'C_D': 'CAN DoS',
    'P_I': 'PTP Sync',
    'M_F': 'Switch MAC Flooding',
    'F_I': 'Frame Injection',
    'C_R': 'CAN Replay',
})

# cap = pyshark.FileCapture(
#     raw_x_path,
#     # keep_packets=False,   # stream, avoids memory blow-up
#     use_json=True         # faster parsing
# )

with PcapReader(raw_x_path) as pcap_reader:
    for pkt in pcap_reader:
        if pkt[Ether].type == 8944:
            print(pkt.time, pkt[Ether].type)
