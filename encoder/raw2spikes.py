from stream.utils import inputFire
from config import INPUT_MAP, WEIGHTS
import struct
from collections import Counter, defaultdict


ETH_LEN = 14
UDP_LARGE = 512
TTL_INC_TH = 1
IP_LEN_BIG = 1000
IP_LEN_SMALL = 60
DEF_MAC_MULTI = 2


NS, CWR, ECE, URG, ACK, PSH, RST, SYN, FIN = (
    1 << 8,
    1 << 7,
    1 << 6,
    1 << 5,
    1 << 4,
    1 << 3,
    1 << 2,
    1 << 1,
    1
)


def init_ttfs_state():
    return {
        "ip_to_macs" : defaultdict(set),
        "ttl_base" : {},
        "seen_tcp_ports" : set(),
        "ip_pref" : Counter(),
        "local" : None
    }


def _weights(feature):
    return WEIGHTS.get(feature, 1)


def _fire(feature, t0_ms, w=None):
    if feature not in INPUT_MAP: return
    weights = w if w is not None else _weights(feature)
    for _ in range(weights):
        inputFire(INPUT_MAP[feature], t0_ms)


def parse_eth(raw):
    if len(raw) < ETH_LEN: return None
    dst, src = raw[:6], raw[6:12]
    etype = struct.unpack("!H", raw[12:14])[0]
    return {
        "dst" : dst,
        "src" : src,
        "etype" : etype,
        "payload" : raw[14:]
    }


def parse_ipv4(raw):
    if len(raw) < 20: return None
    ihl_decl = (raw[0] & 0x0F) * 4
    ihl = 20 if len(raw) < ihl_decl else ihl_decl
    return {
        "ihl" : ihl,
        "ttl" : raw[8],
        "proto" : raw[9],
        "totlen" : struct.unpack("!H", raw[2:4])[0],
        "src" : raw[12:16],
        "dst" : raw[16:20],
        "payload" : raw[ihl:]
    }


def parse_tcp(raw):
    if len(raw) < 16: return None
    sport, dport, seq, ack, off_flags, win = struct.unpack("!HHIIHH", raw[:16])
    doff = (off_flags >> 12) & 0xF
    hdr_len = max(20, min(doff*4, len(raw)))
    flags = off_flags & 0x01FF
    return {
        "sport" : sport,
        "dport" : dport,
        "seq" : seq,
        "ack" : ack,
        "win" : win,
        "flags" : flags,
        "payload_len" : len(raw) - hdr_len
    }


def parse_udp(raw):
    if len(raw) < 8: return None
    sport, dport, ulen = struct.unpack("!HHH", raw[:6])
    return {
        "sport" : sport,
        "dport" : dport,
        "ulen" : ulen
    }


def parse_arp(raw):
    if len(raw) < 8: return None
    op = struct.unpack("!H", raw[6:8])[0]
    sha = raw[8:14] if len(raw) >= 14 else b''
    spa = raw[14:18] if len(raw) >= 18 else b''
    tha = raw[18:24] if len(raw) >= 24 else b''
    tpa = raw[24:28] if len(raw) >= 28 else b''
    return {
        "op" : op,
        "sha" : sha,
        "spa" : spa,
        "tha" : tha,
        "tpa" : tpa
    }


def _ip_multi_mac(state, ip, mac):
    s = state["ip_to_macs"][ip]
    s.add(mac); return len(s) >= DEF_MAC_MULTI


def _update_local_prefix(state, src, dst):
    if state["local"]: return
    for ip in (src, dst):
        state["ip_pref"][ip[:3]] += 1
    if sum(state["ip_pref"].values()) > 3000:
        state["local"], _ = state["ip_pref"].most_common(1)[0]


def encode_packet(ts_s, raw, state):
    t_ms = ts_s * 1000.0
    eth = parse_eth(raw)
    if not eth: return

    # ARP
    if eth["etype"] == 0x0806:
        arp = parse_arp(eth["payload"])
        if not arp: return
        spa, sha = arp["spa"], arp["sha"]

        if _ip_multi_mac(state, ip=spa, mac=sha):
            _fire("ip_multi_mac", t_ms)

        return
    
    # IPv4
    if eth["etype"] != 0x0800:
        return
    ip = parse_ipv4(eth["payload"])
    if not ip: return

    # LAN/TTL
    _update_local_prefix(state, ip["src"], ip["dst"])
    base = state["ttl_base"].setdefault(ip["dst"], ip["ttl"])
    if ip["ttl"] - base >= TTL_INC_TH:
        _fire("ttl_increase", t_ms)
    if ip["ttl"] < base:
        state["ttl_base"][ip["dst"]] = ip["ttl"]

    if ip["totlen"] >= IP_LEN_BIG:
        _fire("ip_totlen_large", t_ms)
    elif ip["totlen"] <= IP_LEN_SMALL:
        _fire("ip_totlen_small", t_ms)

    if _ip_multi_mac(state, ip["src"], eth["src"]):
        _fire("ip_multi_mac", t_ms)


    # TCP
    if ip["proto"] == 6:
        tcp = parse_tcp(ip["payload"])
        if not tcp: return
        F = tcp["flags"]

        if F in (0, NS):
            _fire("tcp_null", t_ms)
        if (F & (FIN|PSH)) == (FIN|PSH) and not (F & SYN):
            _fire("tcp_xmas", t_ms)
        if (F & FIN) and not (F & (SYN|ACK|RST)) and tcp["payload_len"] == 0:
            _fire("tcp_fin_scan", t_ms)

        if (F & SYN) and not (F & ACK):
            _fire("tcp_syn", t_ms)
        if (F & SYN) and (F & ACK):
            _fire("tcp_synack", t_ms)
        if (F & RST) and not (F & ACK):
            _fire("tcp_rst", t_ms)

        tag = (ip["src"], tcp["dport"])
        if tag not in state["seen_tcp_ports"]:
            state["seen_tcp_ports"].add(tag)
            _fire("new_tcp_dport", t_ms)


    # UDP
    elif ip["proto"] == 17:
        udp = parse_udp(ip["payload"])
        if not udp: return
        if udp["ulen"] > UDP_LARGE:
            _fire("udp_len_large", t_ms)

        