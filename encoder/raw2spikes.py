from config import INPUT_MAP
import struct
from collections import Counter, defaultdict


ETH_LEN = 14
UDP_LARGE = 512
TTL_INC_TH = 1
IP_LEN_BIG = 1000
IP_LEN_SMALL = 60
DEF_MAC_MULTI = 2

BCAST_MAC = b"\xff\xff\xff\xff\xff\xff"  
ARP_REQ_THRESHOLD = 20

NS, ACK, PSH, RST, SYN, FIN = (
    1 << 8,
    1 << 4,
    1 << 3,
    1 << 2,
    1 << 1,
    1
)


def init_input_state():
    return {
        "ip_to_macs" : defaultdict(set),
        "ttl_base" : {},
        "seen_tcp_ports" : set(),
        "ip_pref" : Counter(),
        "local" : None,
        "ip_to_macs_arp": defaultdict(set),
        "mac_to_tpas" : defaultdict(set)
    }


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
    if len(raw) < 14: return None
    sport, dport, _, _, off_flags = struct.unpack("!HHIIH", raw[:14])
    doff = (off_flags >> 12) & 0xF
    hdr_len = max(20, min(doff*4, len(raw)))
    flags = off_flags & 0x01FF
    return {
        "sport" : sport,
        "dport" : dport,
        "flags" : flags,
        "payload_len" : len(raw) - hdr_len,
    }


def parse_udp(raw):
    if len(raw) < 8: return None
    _, dport, ulen = struct.unpack("!HHH", raw[:6])
    return {
        "dport" : dport,
        "ulen" : ulen
    }


def parse_arp(raw):
    if len(raw) < 8: return None
    op = struct.unpack("!H", raw[6:8])[0]
    sha = raw[8:14] if len(raw) >= 14 else b''
    spa = raw[14:18] if len(raw) >= 18 else b''
    tpa = raw[24:28] if len(raw) >= 28 else b''
    return {
        "op" : op,
        "sha" : sha,
        "spa" : spa,
        "tpa" : tpa
    }


def _ip_multi_mac(state, ip, mac):
    s = state["ip_to_macs"][ip]
    s.add(mac)
    return len(s) >= DEF_MAC_MULTI


def _update_local_prefix(state, src, dst):
    if state["local"]: return
    for ip in (src, dst):
        state["ip_pref"][ip[:3]] += 1
    if sum(state["ip_pref"].values()) > 3000:
        state["local"], _ = state["ip_pref"].most_common(1)[0]



def _in_id(feature):
    return INPUT_MAP.get(feature, -1)


def encode_packet(ts_s, raw, state):
    t_ms = ts_s * 1000.0
    eth = parse_eth(raw)

    # [(input_id, t_ms), ...]
    spikes = []
    if not eth: return spikes

    # ARP
    if eth["etype"] == 0x0806:
        arp = parse_arp(eth["payload"])
        if not arp: return spikes
        spa, sha, op = arp["spa"], arp["sha"], arp["op"]

        if _ip_multi_mac(state, ip=spa, mac=sha):
            spikes.append((_in_id("ip_multi_mac"), t_ms))

        if arp["spa"] and arp["spa"] == arp["tpa"]:
            spikes.append((_in_id("arp_gratuitous"), t_ms))

        s_dup = state["ip_to_macs_arp"][spa]
        # reply only
        if op == 2:
            s_dup.add(sha)
            if len(s_dup) >= DEF_MAC_MULTI:
                spikes.append((_in_id("arp_dup_ip"), t_ms))

        # request
        if op == 1:
            s_many = state["mac_to_tpas"][sha]
            s_many.add(arp["tpa"])
            if len(s_many) >= ARP_REQ_THRESHOLD:
                spikes.append((_in_id("arp_many_req"), t_ms))

        return spikes
    
    # IPv4
    if eth["etype"] != 0x0800:
        return spikes
    ip = parse_ipv4(eth["payload"])
    if not ip: return spikes

    # LAN/TTL
    _update_local_prefix(state, ip["src"], ip["dst"])
    base = state["ttl_base"].setdefault(ip["dst"], ip["ttl"])
    if ip["ttl"] - base >= TTL_INC_TH:
        spikes.append((_in_id("ttl_increase"), t_ms))
    if ip["ttl"] < base:
        state["ttl_base"][ip["dst"]] = ip["ttl"]

    if ip["totlen"] >= IP_LEN_BIG:
        spikes.append((_in_id("ip_totlen_large"), t_ms))
    elif ip["totlen"] <= IP_LEN_SMALL:
        spikes.append((_in_id("ip_totlen_small"), t_ms))

    if _ip_multi_mac(state, ip["src"], eth["src"]):
        spikes.append((_in_id("ip_multi_mac"), t_ms))


    # TCP
    if ip["proto"] == 6:
        tcp = parse_tcp(ip["payload"])
        if not tcp: return spikes
        F = tcp["flags"]

        if F in (0, NS):
            spikes.append((_in_id("tcp_null"), t_ms))
        if (F & (FIN|PSH)) == (FIN|PSH) and not (F & SYN):
            spikes.append((_in_id("tcp_xmas"), t_ms))
        if (F & FIN) and not (F & (SYN|ACK|RST)) and tcp["payload_len"] == 0:
            spikes.append((_in_id("tcp_fin_scan"), t_ms))

        if (F & SYN) and not (F & ACK):
            spikes.append((_in_id("tcp_syn"), t_ms))

        if (F & SYN) and (F & ACK):
            spikes.append((_in_id("tcp_synack"), t_ms))
        if (F & RST) and not (F & ACK):
            spikes.append((_in_id("tcp_rst"), t_ms))

        tag = (ip["src"], tcp["dport"])
        if tag not in state["seen_tcp_ports"]:
            state["seen_tcp_ports"].add(tag)
            spikes.append((_in_id("new_tcp_dport"), t_ms))


    # UDP
    elif ip["proto"] == 17:
        udp = parse_udp(ip["payload"])
        if not udp: return spikes
        if udp["ulen"] > UDP_LARGE:
            spikes.append((_in_id("udp_len_large"), t_ms))
        
    return spikes

        