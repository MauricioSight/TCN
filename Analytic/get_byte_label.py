import pyshark
from pyshark.packet.packet import Packet
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.fields import LayerField
import subprocess
import json

raw_x_path = "./data/tow-ids-dataset/raw/Automotive_Ethernet_with_Attack_original_10_17_19_50_training.pcap"

cap = pyshark.FileCapture(
    raw_x_path,
    include_raw = True,
    use_ek=True         # faster parsing
    # keep_packets=False,   # stream, avoids memory blow-up
    # debug=True,
)

def get_field(field: LayerField):
    a = field.pos

def get_layer(layer: JsonLayer):
    field_names = layer.field_names
    for name in field_names:
        get_field(layer.get(name))

def get_pkt(pkt: Packet):
    raw_packet = pkt.get_raw_packet()
    if "ieee1722" in pkt:
        rec_raw_pkt = []
        for layer in pkt.layers:
            get_layer(layer)

for pkt in cap:
    get_pkt(pkt)
