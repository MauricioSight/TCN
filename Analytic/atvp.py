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

cap = pyshark.FileCapture(
    raw_x_path,
    # keep_packets=False,   # stream, avoids memory blow-up
    use_json=True         # faster parsing
)

all_data = []

for pkt in cap:
    data = {**pkt.frame_info._all_fields}
    for l in pkt.layers:
        data = {**data, **l._all_fields}

    keys_to_remove = []
    for key in data.keys():
        if '_tree' in key:
            keys_to_remove.append(key)
    for k in keys_to_remove:
        del data[k]
    
    if "ieee1722" in pkt:
        if 'iec61883.videodata' in data:
            del data['iec61883.videodata']
        if 'iec61883.videodata_tree' in data:
            del data['iec61883.videodata_tree']

    elif 'ptp' in pkt:
        for key in data.keys():
            if 'correctionField' in key:
                data = {**data, **data[key]}
                del data[key]
                break
        if '_ws.expert' in data:
            del data['_ws.expert']
        if 'Follow Up information TLV' in data:
            del data['Follow Up information TLV']

    elif 'udp' in pkt:
        del data['udp.port']
        if 'ip.host' in data:
            del data['ip.host']
            del data['ip.addr']
        if 'ipv6.host' in data:
            del data['ipv6.host']
            del data['ipv6.addr']
        data = {**data, **data['Timestamps']}
        del data['Timestamps']

    elif 'mrp-mvrp' in pkt:
        del data['mrp-mvrp.message']

    elif 'mrp-msrp' in pkt:
        del data['mrp-msrp.message']

    else:
        pass

    all_data.append(data)

df = pd.DataFrame(all_data)
df.to_csv('./avtp_packets.csv', index=False)