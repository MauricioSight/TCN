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
#     keep_packets=False,   # stream, avoids memory blow-up
#     use_json=True         # faster parsing
# )

def get(layer, attr):
    """Safely extract field values"""
    try:
        return getattr(layer, attr)
    except Exception:
        return None

def extract_edges_all(pcap, max_packets=None):
    cap = pyshark.FileCapture(pcap, keep_packets=False, use_json=True)
    rows = []

    for i, pkt in tqdm(enumerate(cap), total=1203737):
        try:
            # Default values
            src, dst, proto, extra = None, None, None, None

            # Ethernet-level
            if "eth" in pkt:
                src, dst = pkt.eth.src, pkt.eth.dst

            # IP-level
            if "ip" in pkt:
                src, dst = pkt.ip.src, pkt.ip.dst
                proto = f"IP/{pkt.ip.proto}"   # proto is numeric (e.g., 17 for UDP)

            # UDP/TCP
            if "udp" in pkt:
                src = f"{pkt.ip.src}:{pkt.udp.srcport}"
                dst = f"{pkt.ip.dst}:{pkt.udp.dstport}"
                proto = "UDP"
            elif "tcp" in pkt:
                src = f"{pkt.ip.src}:{pkt.tcp.srcport}"
                dst = f"{pkt.ip.dst}:{pkt.tcp.dstport}"
                proto = "TCP"

            # ARP
            elif "arp" in pkt:
                src = pkt.arp.src_proto_ipv4
                dst = pkt.arp.dst_proto_ipv4
                proto = "ARP"

            # PTP
            elif "ptp" in pkt:
                proto = "PTP"
                extra = f"msgid={getattr(pkt.ptp, 'messageid', None)}"

            # IEEE1722 / AVTP
            elif "ieee1722" in pkt:
                proto = "IEEE1722"
                extra = getattr(pkt.IEC61883, "stream_id", None)

            # IEEE17221 (AVDECC)
            elif "ieee17221" in pkt:
                proto = "IEEE17221"

            # DHCP (bootp dissector)
            elif "bootp" in pkt:
                proto = "DHCP"

            # mDNS
            elif "mdns" in pkt:
                proto = "mDNS"

            # fallback: generic
            if proto is None:
                proto = pkt.highest_layer

            # key = f'[{proto}]{src}->{dst}'
            # if key not in edges:
            #     edges[key] = {
            #         'Normal': 0,
            #         'CAN DoS': 0,
            #         'PTP Sync': 0,
            #         'Switch MAC Flooding': 0,
            #         'Frame Injection': 0,
            #         'CAN Replay': 0,
            #     }

            # label = labels.iloc[i]
            # edges[key][label['label']] += 1
            # edges[key]['extra'] = extra or ''

        except Exception as e:
            pass

        try:
            # Base fields
            time_epoch = float(pkt.sniff_timestamp)
            length     = int(get(pkt, "length")) if hasattr(pkt, "length") else None
            protocols  = get(pkt.frame_info, "protocols")

            # Ethernet
            eth_src = get(pkt.eth, "src") if "eth" in pkt else None
            eth_dst = get(pkt.eth, "dst") if "eth" in pkt else None
            eth_type = get(pkt.eth, "type") if "eth" in pkt else None

            # VLAN
            vlan_id = get(pkt.vlan, "id") if "vlan" in pkt else None

            # IP
            ip_src = get(pkt.ip, "src") if "ip" in pkt else None
            ip_dst = get(pkt.ip, "dst") if "ip" in pkt else None
            ip_proto = get(pkt.ip, "proto") if "ip" in pkt else None

            # TCP/UDP
            tcp_sport = get(pkt.tcp, "srcport") if "tcp" in pkt else None
            tcp_dport = get(pkt.tcp, "dstport") if "tcp" in pkt else None
            udp_sport = get(pkt.udp, "srcport") if "udp" in pkt else None
            udp_dport = get(pkt.udp, "dstport") if "udp" in pkt else None

            # IEEE1722 / AVTP
            avtp_verfield           = get(pkt.IEEE1722, "verfield") if "IEEE1722" in pkt else None
            avtp_subtype            = get(pkt.IEEE1722, "subtype") if "IEEE1722" in pkt else None
            avtp_svfield            = get(pkt.IEEE1722, "svfield") if "IEEE1722" in pkt else None

            avtp_dbs                = get(pkt.IEC61883, "dbs") if "IEC61883" in pkt else None
            avtp_sph                = get(pkt.IEC61883, "sph") if "IEC61883" in pkt else None
            avtp_dbc                = get(pkt.IEC61883, "dbc") if "IEC61883" in pkt else None
            avtp_channel            = get(pkt.IEC61883, "channel") if "IEC61883" in pkt else None
            avtp_sy                 = get(pkt.IEC61883, "sy") if "IEC61883" in pkt else None
            avtp_tcode              = get(pkt.IEC61883, "tcode") if "IEC61883" in pkt else None
            avtp_sid                = get(pkt.IEC61883, "sid") if "IEC61883" in pkt else None
            avtp_qi2                = get(pkt.IEC61883, "qi2") if "IEC61883" in pkt else None
            avtp_qi1                = get(pkt.IEC61883, "qi1") if "IEC61883" in pkt else None
            avtp_fn                 = get(pkt.IEC61883, "fn") if "IEC61883" in pkt else None
            avtp_tag                = get(pkt.IEC61883, "tag") if "IEC61883" in pkt else None
            avtp_tufield            = get(pkt.IEC61883, "tufield") if "IEC61883" in pkt else None
            avtp_tvfield            = get(pkt.IEC61883, "tvfield") if "IEC61883" in pkt else None
            avtp_fdf_tsf            = get(pkt.IEC61883, "fdf_tsf") if "IEC61883" in pkt else None
            avtp_qpc                = get(pkt.IEC61883, "qpc") if "IEC61883" in pkt else None
            avtp_seqnum             = get(pkt.IEC61883, "seqnum") if "IEC61883" in pkt else None
            avtp_mrfield            = get(pkt.IEC61883, "mrfield") if "IEC61883" in pkt else None
            avtp_avtp_timestamp     = get(pkt.IEC61883, "avtp_timestamp") if "IEC61883" in pkt else None
            avtp_gvfield            = get(pkt.IEC61883, "gvfield") if "IEC61883" in pkt else None
            avtp_stream_data_len    = get(pkt.IEC61883, "stream_data_len") if "IEC61883" in pkt else None
            avtp_stream_id          = get(pkt.IEC61883, "stream_id") if "IEC61883" in pkt else None
            avtp_fdf_no_syt         = get(pkt.IEC61883, "fdf_no_syt") if "IEC61883" in pkt else None
            avtp_fmt                = get(pkt.IEC61883, "fmt") if "IEC61883" in pkt else None

            rows.append({
                "time_epoch": time_epoch,
                "frame.len": length,
                "protocols": protocols,
                "eth.src": eth_src, "eth.dst": eth_dst, "eth.type": eth_type,
                "vlan.id": vlan_id,
                "ip.src": ip_src, "ip.dst": ip_dst, "ip.proto": ip_proto,
                "tcp.srcport": tcp_sport, "tcp.dstport": tcp_dport,
                "udp.srcport": udp_sport, "udp.dstport": udp_dport,
                "src": src,
                "dst": dst,
                "proto": proto,
                "extra": extra,
                'avtp_verfield': avtp_verfield,
                'avtp_subtype': avtp_subtype,
                'avtp_svfield': avtp_svfield,
                'avtp_dbs': avtp_dbs,
                'avtp_sph': avtp_sph,
                'avtp_dbc': avtp_dbc,
                'avtp_channel': avtp_channel,
                'avtp_sy': avtp_sy,
                'avtp_tcode': avtp_tcode,
                'avtp_sid': avtp_sid,
                'avtp_qi2': avtp_qi2,
                'avtp_qi1': avtp_qi1,
                'avtp_fn': avtp_fn,
                'avtp_tag': avtp_tag,
                'avtp_tufield': avtp_tufield,
                'avtp_tvfield': avtp_tvfield,
                'avtp_fdf_tsf': avtp_fdf_tsf,
                'avtp_qpc': avtp_qpc,
                'avtp_seqnum': avtp_seqnum,
                'avtp_mrfield': avtp_mrfield,
                'avtp_avtp_timestamp': avtp_avtp_timestamp,
                'avtp_gvfield': avtp_gvfield,
                'avtp_stream_data_len': avtp_stream_data_len,
                'avtp_stream_id': avtp_stream_id,
                'avtp_fdf_no_syt': avtp_fdf_no_syt,
                'avtp_fmt': avtp_fmt,
            })
        except Exception as e:
            print(f"Error parsing packet {i}: {e}")
            rows.append({})
            pass

    cap.close()
    return rows

rows = extract_edges_all(raw_x_path)
df = pd.DataFrame(rows)
df.to_csv('./packets.csv', index=False)
