import dpkt
import numpy as np
from stream.utils import inputFire
from config import INPUT_MAP


PACKET_SIZE = (6, 7, 8)
TH_SMALL, TH_BIG = 200, 800



def init_ttfs_state(dmax_ms=20.0):
    return {
        "dmax_ms" : dmax_ms,
        "max_size" : 200.0
    }


def encode_packet(
        ts_s,
        raw_bytes,
        state
    ):
    t0_ms = ts_s * 1000.0

    eth = dpkt.ethernet.Ethernet(raw_bytes, unpack=False)
    etype = eth.type

    # ARP
    if etype == dpkt.ethernet.ETH_TYPE_ARP:
        # 1 - request, 2 - reply
        op = eth.arp.op
        input_id = INPUT_MAP["arp_req"] if op == 1 else INPUT_MAP["arp_rep"]
        inputFire(input_id, t0_ms)
    
    # IPv4 (TCP/UDP)
    elif etype == dpkt.ethernet.ETH_TYPE_IP:
        ip = eth.data
        if ip.p == dpkt.ip.IP_PROTO_TCP:
            tcp = ip.data
            flags = tcp.flags
            if flags & dpkt.tcp.TH_SYN and not flags & dpkt.tcp.TH_ACK:
                inputFire(INPUT_MAP["tcp_syn"], t0_ms)
            elif flags & dpkt.tcp.TH_SYN and not flags & dpkt.tcp.TH_ACK:
                inputFire(INPUT_MAP["tcp_synack"], t0_ms)
            elif flags & dpkt.tcp.TH_RST:
                inputFire(INPUT_MAP["tcp_rst"], t0_ms)
            else:
                inputFire(INPUT_MAP["other_tcp"], t0_ms)
        elif ip.p == dpkt.ip.IP_PROTO_UDP:
            udp = ip.data
            # b'\xef\xff\xff\xfa' = 239.255.255.250
            if udp.dport == 1900 or ip.dst == b'\xef\xff\xff\xfa':
                inputFire(INPUT_MAP["ssdp"], t0_ms)
            else:
                inputFire(INPUT_MAP["other_udp"], t0_ms)
    
    # TTFS
    size = len(raw_bytes)
    max_size_prev = state["max_size"]
    max_size_cur = 0.9 * max_size_prev + 0.1 * size
    state["max_size"] = max_size_cur

    delay_ms = state["dmax_ms"] * (1.0 - np.tanh(size / max_size_cur))

    if size >= TH_BIG:
        psize = PACKET_SIZE[0]
    elif size <= TH_SMALL:
        psize = PACKET_SIZE[2]
    else:
        psize = PACKET_SIZE[1]

    inputFire(psize, t0_ms + delay_ms)


    

