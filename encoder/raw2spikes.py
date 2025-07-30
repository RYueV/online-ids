from config import INPUT_MAP
import struct
from collections import Counter, defaultdict


NS, URG, ACK, PSH, RST, SYN, FIN = (
    1 << 8,
    1 << 5,
    1 << 4,
    1 << 3,
    1 << 2,
    1 << 1,
    1
)

# ttl/hop increase
INC_TH = 2

# ipv4
UDP_LARGE = 1500
IP_LEN_BIG = 1500
IP_LEN_SMALL = 40
UDP_HIGH_SPORT_TH = 57000

# arp
DEF_MAC_MULTI = 2
ARP_REQ_THRESHOLD = 5
MAX_SYN_TRACK = 10000

# ipv6
IP6_PLEN_SMALL = 80
HOP_LOW_TH = 12
HOP_HIGH_TH = 160



def init_input_state():
    return {
        "ip_to_macs" : defaultdict(set),
        "ttl_base" : {},
        "seen_tcp_ports" : set(),
        "ip_pref" : Counter(),
        "local" : None,
        "ip_to_macs_arp": defaultdict(set),
        "mac_to_tpas" : defaultdict(set),
        "syn_initiated": set(),
        "hop_base": {}
    }


def parse_eth(raw):
    if len(raw) < 14: return None
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
    flags_off = struct.unpack("!H", raw[6:8])[0]
    flags = flags_off >> 13
    frag_offset = flags_off & 0x1FFF
    return {
        "ihl" : ihl,
        "ttl" : raw[8],
        "proto" : raw[9],
        "totlen" : struct.unpack("!H", raw[2:4])[0],
        "src" : raw[12:16],
        "dst" : raw[16:20],
        "payload" : raw[ihl:],
        "flags": flags,
        "frag_offset": frag_offset
    }


def parse_ipv6(raw):
    if len(raw) < 40: return None
    _, _, nexth, hop = struct.unpack("!IHBB", raw[:8])
    src, dst = raw[8:24], raw[24:40]
    return {
        "next": nexth,
        "hop": hop,
        "src": src,
        "dst": dst,
        "payload": raw[40:]
    }



def parse_tcp(raw):
    if len(raw) < 14: return None
    sport, dport, _, _, off_flags = struct.unpack("!HHIIH", raw[:14])
    doff = (off_flags >> 12) & 0xF
    hdr_len = max(20, min(doff*4, len(raw)))
    flags = off_flags & 0x01FF
    window = struct.unpack("!H", raw[14:16])[0] if len(raw) >= 16 else 0
    return {
        "sport" : sport,
        "dport" : dport,
        "flags" : flags,
        "data_offset": doff,
        "window": window
    }


def parse_udp(raw):
    if len(raw) < 8: return None
    sport, _, ulen = struct.unpack("!HHH", raw[:6])
    return {
        "ulen" : ulen,
        "sport": sport
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
    spikes = []             # [(input_id, t_ms), ...]

    eth = parse_eth(raw)
    if not eth: 
        spikes.append((_in_id("generic_pkt"), t_ms))
        return spikes

    # ARP
    if eth["etype"] == 0x0806:
        arp = parse_arp(eth["payload"])
        if arp: 
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
    

    # IPv4
    elif eth["etype"] == 0x0800:
        ip = parse_ipv4(eth["payload"])
        if not ip: 
            if not spikes: spikes.append((_in_id("generic_pkt"), t_ms))
            return spikes

        # ttl anomaly
        _update_local_prefix(state, ip["src"], ip["dst"])
        base = state["ttl_base"].setdefault(ip["dst"], ip["ttl"])
        if ip["ttl"] - base >= INC_TH:
            spikes.append((_in_id("ttl_increase"), t_ms))
        if ip["ttl"] < base:
            state["ttl_base"][ip["dst"]] = ip["ttl"]
        if ip["ttl"] <= 5:
            spikes.append((_in_id("ip_ttl_low"), t_ms))
        elif ip["ttl"] >= 200:
            spikes.append((_in_id("ip_ttl_high"), t_ms))

        # ip length
        if ip["totlen"] >= IP_LEN_BIG:
            spikes.append((_in_id("ip_totlen_large"), t_ms))
        elif ip["totlen"] <= IP_LEN_SMALL:
            spikes.append((_in_id("ip_totlen_small"), t_ms))

        if _ip_multi_mac(state, ip["src"], eth["src"]):
            spikes.append((_in_id("ip_multi_mac"), t_ms))

        # TCP
        if ip["proto"] == 6:
            if ip["frag_offset"] != 0: tcp = None
            else: tcp = parse_tcp(ip["payload"][:20])
            if tcp is None: 
                if not spikes: spikes.append((_in_id("generic_pkt"), t_ms))
                return spikes

            F = tcp["flags"]
            syn_key = (ip["src"], ip["dst"], tcp["sport"], tcp["dport"])
            rev_key = (ip["dst"], ip["src"], tcp["dport"], tcp["sport"])

            if (F & SYN) and not (F & ACK):
                state["syn_initiated"].add(syn_key)
                if len(state["syn_initiated"]) > MAX_SYN_TRACK:
                    state["syn_initiated"].pop()
            elif (F & SYN) and (F & ACK):
                if rev_key in state["syn_initiated"]:
                    state["syn_initiated"].remove(rev_key)
                else:
                    spikes.append((_in_id("tcp_unexpected_synack"), t_ms))
            elif (F & ACK) and not (F & SYN) and not (F & RST) and not (F & FIN):
                if syn_key in state["syn_initiated"]:
                    state["syn_initiated"].remove(syn_key)

            if (F & SYN) and not (F & ACK):
                spikes.append((_in_id("tcp_syn"), t_ms))
            if (F & SYN) and (F & ACK):
                spikes.append((_in_id("tcp_synack"), t_ms))
            if (F & SYN) and tcp["data_offset"] > 5:
                spikes.append((_in_id("tcp_data_offset_opts"), t_ms))
            if (F & ACK) and not (F & (SYN | FIN | RST | PSH | URG)):
                spikes.append((_in_id("tcp_ack_only"), t_ms))
            if (F & ACK) and (F & PSH) and not (F & (SYN | FIN | RST | URG)):
                spikes.append((_in_id("tcp_ack_psh"), t_ms))
            if (F & ACK) and (F & RST):
                spikes.append((_in_id("tcp_ack_rst"), t_ms))

            if tcp["window"] == 0 and (F & ACK):
                spikes.append((_in_id("tcp_win_zero_ack"), t_ms))

            tag = (ip["src"], tcp["dport"])
            if tag not in state["seen_tcp_ports"]:
                state["seen_tcp_ports"].add(tag)
                spikes.append((_in_id("new_tcp_dport"), t_ms))

        # UDP
        elif ip["proto"] == 17:
            udp = parse_udp(ip["payload"])
            if udp: 
                if udp["ulen"] > UDP_LARGE:
                    spikes.append((_in_id("udp_len_large"), t_ms))
                if udp["sport"] >= UDP_HIGH_SPORT_TH:
                    spikes.append((_in_id("udp_sport_high"), t_ms)) 


    # ipv6
    elif eth["etype"] == 0x86DD:
        ip6 = parse_ipv6(eth["payload"]) 
        if not ip6: return spikes

        key = ip6["src"]
        base = state["hop_base"].setdefault(key, ip6["hop"])

        if ip6["hop"] - base >= INC_TH:
            spikes.append((_in_id("hop_increase"), t_ms))
        if ip6["hop"] < base:
            state["hop_base"][key] = ip6["hop"]

        if ip6["hop"] <= HOP_LOW_TH:
            spikes.append((_in_id("ip6_hop_low"), t_ms))
        elif ip6["hop"] >= HOP_HIGH_TH:
            spikes.append((_in_id("ip6_hop_high"), t_ms))

        # tcp
        if ip6["next"] == 6:
            tcp = parse_tcp(ip6["payload"])
            if tcp:
                F = tcp["flags"]
                if (F & SYN) and not (F & ACK):
                    spikes.append((_in_id("tcp_syn"), t_ms))
                if (F & ACK) and not (F & (SYN | FIN | RST)):
                    spikes.append((_in_id("tcp_ack_only"), t_ms))
        
        # udp
        elif ip6["next"] == 17:
            udp = parse_udp(ip6["payload"])
            if udp:
                if udp["ulen"] > UDP_LARGE:
                    spikes.append((_in_id("udp_len_large"), t_ms))
                if udp["sport"] >= UDP_HIGH_SPORT_TH:
                    spikes.append((_in_id("udp_sport_high"), t_ms))

    if not spikes:
        spikes.append((_in_id("generic_pkt"), t_ms))
    
    return spikes

        