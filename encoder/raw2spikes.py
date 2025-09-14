import struct
from collections import defaultdict, deque
from math import log10
from config import INPUT_MAP


# ttl/hop бины для фоновых сигналов
TTL_BINS = [15, 40, 50, 62, 70]
# HOP_BINS = [10, 24, 48, 72, 96] 

# отклонение ttl/hop от базового для "hop/ttl increase"
INC_TH_TTL = 4
# INC_TH_HOP = 2

# логарифмические бины (степени 10ки) для window
WIN_POWS  = [2.3, 2.7, 3.0, 3.3, 3.7, 4.0, 4.3, 4.8]
# для dseq/dack
DSEQ_POWS = [2.5, 2.8, 3.0, 3.2, 3.5, 3.8]

# линейные бины для полей, связанных с длиной IP/UDP 
# (той, что указана в заголовке; не физической)
LEN_BINS = [32, 256, 512, 1024, 1128, 1256, 1500]

# MSS бины (полезная нагрузка в байтах)  
MSS_BINS = [536, 1200, 1460]

# размер полезной нагрузки в байтах для ICMP протокола
ICMP_PL_BINS = [0, 8, 32, 64, 128, 256]

# интервал ttfs-кодера внутри своего канала в мс
TTFS_FRAME_MS = 2.0

# интервал в с для "tcp_syn_no_ack"
SYN_NOACK_TO = 1.8
# интервал в с для "tcp_syn_retransmit"
# SYN_RETRANS_TO = 0.2
# макс. win для "tcp_small_win"
SYN_SMALL_WIN_TH = 1024

# порог срабатывания "arp_many_req"
ARP_REQ_TH = 7

# список подозрительных портов
SUS_UDP_PORTS = {1900, 53413, 3702, 5000, 4444, 53, 123, 161}
SUS_TCP_PORTS = {23, 2323, 81, 7547, 5555, 21}

# fragment overlap & tiny fragments
TINY_FRAG_PL_TH = 24 
MAX_SYN_TRACK = 10000

# низкоуровневые парсеры
ETH_HDR = struct.Struct("!6s6sH")
TCP_MIN = struct.Struct("!HHIIH")   # up to data offset + flags
UDP_MIN = struct.Struct("!HHH")
ICMP_MIN = struct.Struct("!BBH")    # type, code, checksum

# TCP-флаги
NS, CWR, ECE, URG, ACK, PSH, RST, SYN, FIN = (1<<8, 1<<7, 1<<6, 1<<5, 1<<4, 1<<3, 1<<2, 1<<1, 1<<0)



# возвращает id признака по его имени
def _in_id(feature: str)->int:
    return INPUT_MAP.get(feature, -1)


### номер бина из bins, куда попадает val 
# (для линейных бинов)
def _pop_linear(val:int, bins):
    for i, b in enumerate(bins):
        if val < b:
            return i
    return len(bins)

# (для логарифмических)
def _pop_log(val:int, pows):
    log_val = log10(max(val, 1))
    for i, p in enumerate(pows):
        if log_val < p:
            return i
    return len(pows)
###


### TTFS
def _ttfs_norm_linear(val:int, bins, idx:int)->float:
    if idx >= len(bins):
        return 0.999
    v = float(val)
    low = 0.0 if idx<=0 else float(bins[idx-1])
    high = float(bins[idx])
    if high <= low: return 0.5
    if v <= low: return 0.0
    if v >= high: return 0.999
    return (v - low) / (high - low)

def _ttfs_norm_log(val:int, pows, idx:int)->float:
    if idx >= len(pows):
        return 0.999
    v = log10(max(int(val), 1))
    low = 0.0 if idx<=0 else float(pows[idx-1])
    high = float(pows[idx])
    if high <= low: return 0.5
    if v <= low: return 0.0
    if v >= high: return 0.999
    return (v - low) / (high - low)

def _ttfs_apply(t_ms:float, norm:float)->float:
    ttfs_t_ms = TTFS_FRAME_MS * max(0.0, min(0.999, float(norm)))
    return t_ms + ttfs_t_ms
###


### определение типа MAC-адреса
def _is_broadcast_mac(mac:bytes)->bool:
    return mac == b"\xff\xff\xff\xff\xff\xff"

def _is_multicast_mac(mac:bytes)->bool:
    return bool(mac[0] & 0x01) and not _is_broadcast_mac(mac)

def _is_local_admin_mac(mac:bytes)->bool:
    return bool(mac[0] & 0x02)
###


### Парсеры протоколов различных уровней OSI
# парсер Ethernet  (L2)     
def _parse_eth(buf:bytes):
    # минимальная длина заголовка 14 байт 
    # (dst mac, src mac, ethertype)
    if len(buf) < 14:
        return None
    dst, src, etype = ETH_HDR.unpack_from(buf)
    off = 14    # смещение на 14 байт
    vlan_id = None
    vlan_depth = 0
    # VLAN-теги, если есть
    while etype in (0x8100, 0x88A8):
        if len(buf) < off + 4:
            return None
        tci, etype = struct.unpack_from("!HH", buf, off)
        vlan_id = tci & 0x0FFF
        vlan_depth += 1
        off += 4
    payload = buf[off:]
    return etype, src, dst, payload, vlan_id, vlan_depth

# парсер IPv4 (L3)
def _parse_ipv4(buf:bytes):
    # длина заголовка в байтах (минимум 20)
    ihl = (buf[0] & 0x0F) * 4 or 20
    if len(buf) < ihl:
        return None
    ttl = buf[8]
    proto = buf[9]
    totlen = struct.unpack_from("!H", buf, 2)[0]
    ident = struct.unpack_from("!H", buf, 4)[0]
    flags_frag = struct.unpack_from("!H", buf, 6)[0]
    flags = (flags_frag & 0xE000) >> 13
    df = bool(flags & 0x2)
    mf = bool(flags & 0x1)
    frag_off = flags_frag & 0x1FFF
    src, dst = buf[12:16], buf[16:20]
    return ttl, proto, totlen, ident, src, dst, df, mf, frag_off, ihl, buf[ihl:]

# парсер IPv6 (L3)
def _parse_ipv6(buf:bytes):
    # минимальная длина заголовка
    if len(buf) < 40:
        return None
    payload_len = struct.unpack_from("!H", buf, 4)[0]
    nxt = buf[6]
    hop = buf[7]
    src = buf[8:24]
    dst = buf[24:40]
    return hop, nxt, payload_len, src, dst, buf[40:]

# парсер TCP (L4)
def _parse_tcp(buf:bytes):
    if len(buf) < 20:
        return None
    sport, dport, seq, ack, off_flags = TCP_MIN.unpack_from(buf)
    data_off = (off_flags >> 12) * 4
    flags = off_flags & 0x01FF
    if len(buf) < data_off:
        data_off = 20
    win = struct.unpack_from("!H", buf, 14)[0]
    opts = buf[20:data_off] if data_off > 20 else b""
    return sport, dport, seq, ack, flags, win, data_off, opts

# парсер UDP (L4)
def _parse_udp(buf:bytes):
    if len(buf) < 8:
        return None
    sport, dport, ulen = UDP_MIN.unpack_from(buf)
    return sport, dport, ulen

# парсер ARP (L2)
def _parse_arp(buf:bytes):
    if len(buf) < 28:
        return None
    op = struct.unpack_from("!H", buf, 6)[0]
    sha = buf[8:14]
    spa = buf[14:18]
    tpa = buf[24:28]
    return op, sha, spa, tpa

# парсер ICMP (L3)
def _parse_icmp(buf:bytes):
    if len(buf) < 4:
        return None
    typ, code, _ = ICMP_MIN.unpack_from(buf)
    ident = seq = None
    if typ in (0, 8) and len(buf) >= 8:
        ident, seq = struct.unpack_from("!HH", buf, 4)
    return typ, code, ident, seq
###


# инициализация состояния входного слоя
def init_input_state():
    return{
        # базовые значения ttl/hop
        "ttl_base": {},
        # "hop_base": {},
        # sequence/ack tracking для четверок (src_ip, dst_ip, src_port, dst_port)
        "last_seq": {},
        "last_ack": {},
        # tcp-handshake и SYN bookkeeping
        # "syn_state": {},
        "syn_track": {},            # (src, dst, sport, dport) -> ts_s of SYN
        "handshake_state": {},      # (src, dst, sport, dport) -> state str
        "fin_sent": set(),
        # ARP
        "arp_pending": {},          # tpa -> ts_s of last who-was
        "ip2mac": defaultdict(set),
        "mac2tpa": defaultdict(set),
        # встреченные tcp/udp порты
        "ip_ports_tcp": defaultdict(set),
        "ip_ports_udp": defaultdict(set),
        # отслеживание порядка фрагментов (ключ - (src, dst, ident, proto))
        "frag_last_end": {},
        # очередь последних использованных ключей для ограничения размера syn_track
        "_syn_order": deque(maxlen=MAX_SYN_TRACK)
    }


# генерация "спайка"
def _push(
        spikes,     # список спайков
        name,       # наименование сработавшего признака
        t_ms        # момент времени в мс
    ):
    fid = _in_id(feature=name)
    if fid >= 0:
        spikes.append((fid, t_ms))


# извлечение бинарных признаков TCP
def _handle_tcp(
        src,        # ip источника
        dst,        # ip получателя
        buf,        # payload IP-пакета, начиная с TCP заголовка
        ts_s,       # время прихода пакета в секундах
        t_ms,       # время прихода пакета в миллисекундах
        spikes,     # список спайков
        st          # словарь состояния
    ):
    # парсинг заголовка
    res = _parse_tcp(buf)
    if not res: return
    # sport, dport - порты источника и назначения;
    # seq - номер первого байта в этом сегменте;
    # ack - следующий ожидаемый байт; flg - набор флагов;
    # win - размер окна приема; doff - длина tcp-заголовка;
    # ops - tcp-опции
    sport, dport, seq, ack, flg, win, doff, opts = res
    
    four = (src, dst, sport, dport)         # прямое направление (клиент->сервер)
    rev_four = (dst, src, dport, sport)     # обратное направление (сервер->клиент)

    # ttfs-кодирование значения win
    bin_win = _pop_log(max(win,1), WIN_POWS)
    norm_win = _ttfs_norm_log(max(win,1), WIN_POWS, bin_win)
    _push(spikes, f"raw_win_{bin_win}", _ttfs_apply(t_ms, norm_win))

    # смещение (прирост) seq/ack между двумя пакетами в одном потоке
    dseq = (seq - st["last_seq"].get(four, seq)) & 0xFFFFFFFF
    dack = (ack - st["last_ack"].get(four, ack)) & 0xFFFFFFFF
    st["last_seq"][four] = seq
    st["last_ack"][four] = ack

    if dseq == 0:
        # sequence number не изменился
        _push(spikes, "raw_dseq_zero", t_ms)
    elif dseq == 1:
        # только управляющий байт
        _push(spikes, "raw_dseq_one", t_ms)
    else:
        bin_dseq = _pop_log(dseq, DSEQ_POWS)
        norm_dseq = _ttfs_norm_log(dseq, DSEQ_POWS, bin_dseq)
        _push(spikes, f"raw_dseq_{bin_dseq}", _ttfs_apply(t_ms, norm_dseq))
    
    if dack == 0:
        # duplicate ACK
        _push(spikes, "raw_dack_zero", t_ms)
    elif dack == 1:
        _push(spikes, "raw_dack_one", t_ms)
    else:
        bin_dack = _pop_log(dack, DSEQ_POWS)
        norm_dack = _ttfs_norm_log(dack, DSEQ_POWS, bin_dack)
        _push(spikes, f"raw_dack_{bin_dack}", _ttfs_apply(t_ms, norm_dack))

    # флаги -> события
    if flg & SYN and not flg & ACK:
        _push(spikes, "flag_syn", t_ms)
    if flg & SYN and flg & ACK:
        _push(spikes, "flag_synack", t_ms)
    if flg & RST:
        _push(spikes, "flag_rst", t_ms)
    if flg & ACK and flg & RST:
        _push(spikes, "flag_ack_rst", t_ms)
    if flg & FIN:
        _push(spikes, "flag_fin", t_ms)
    if flg & URG:
        _push(spikes, "flag_urg", t_ms)
    if flg & ECE:
        _push(spikes, "flag_ece", t_ms)
    if flg & CWR:
        _push(spikes, "flag_cwr", t_ms)

    # scan detection
    if flg == 0:
        _push(spikes, "tcp_null_flags", t_ms)
    if (flg & (FIN | PSH | URG)) == (FIN | PSH | URG) and (flg & ~(FIN | PSH | URG)) == 0:
        _push(spikes, "tcp_xmas_flags", t_ms)
    if (flg & SYN) and (flg & FIN):
        _push(spikes, "tcp_syn_fin_both", t_ms)

    # если длина заголовка больше 20 байт, есть tcp-опции
    if doff > 20:
        _push(spikes, "tcp_opts_present", t_ms)
        i = 0
        while i < len(opts):
            kind = opts[i]
            # end of options list
            if kind == 0: break
            # no operation (NOP)
            if kind == 1:
                i += 1
                continue
            if (i + 1) >= len(opts): break
            length = opts[i+1]
            if length < 2 or i + length > len(opts): break
            data = opts[i+2:i+length]
            # MSS (maximum segment size)
            if kind == 2 and len(data) >= 2:
                mss = struct.unpack("!H", data[:2])[0]
                bin_mss = _pop_linear(mss, MSS_BINS)
                norm_mss = _ttfs_norm_linear(mss, MSS_BINS, bin_mss)
                _push(spikes, f"tcp_opt_mss_{bin_mss}", _ttfs_apply(t_ms, norm_mss))
            # window scale
            elif kind == 3 and len(data) >= 1:
                _push(spikes, "tcp_opt_wscale", t_ms)
            # SACK permitted
            elif kind == 4:
                _push(spikes, "tcp_opt_sack_perm", t_ms)
            # timestamps
            elif kind == 8 and len(data) >= 8:
                _push(spikes, "tcp_opt_timestamp", t_ms)
            i += length

    # новый src-порт -> событие
    if sport not in st["ip_ports_tcp"][src]:
        if st["ip_ports_tcp"][src]:
            _push(spikes, "ip_extended_ports_tcp", t_ms)
        st["ip_ports_tcp"][src].add(sport)

    # подозрительный порт -> событие
    if dport in SUS_TCP_PORTS or sport in SUS_TCP_PORTS:
        _push(spikes, "tcp_port_suspicious", t_ms)

    # handshake tracking
    if flg & SYN and not flg & ACK:

        st["syn_track"][four] = ts_s
        st["_syn_order"].append(four)
        if len(st["syn_track"]) > MAX_SYN_TRACK:
            old = st["_syn_order"].popleft()
            st["syn_track"].pop(old, None)

        st["handshake_state"][four] = "SYN_SENT"
        if win < SYN_SMALL_WIN_TH:
            # маленькое окно приема
            _push(spikes, "tcp_syn_small_win", t_ms)

    elif flg & SYN and flg & ACK:
        state = st["handshake_state"].get(rev_four)
        if state == "SYN_SENT":
            st["handshake_state"][rev_four] = "SYNACK_SEEN"
        else:
            _push(spikes, "tcp_unexpected_synack", t_ms)

    elif flg & ACK and not flg & (SYN | RST | FIN):
        state = st["handshake_state"].get(four)
        if state == "SYNACK_SEEN":
            st["handshake_state"][four] = "ESTABLISHED"
        st["syn_track"].pop(rev_four, None)

    # полуоткрытые соединения
    expired = []
    for k, t0 in list(st["syn_track"].items()):
        if ts_s - t0 > SYN_NOACK_TO:
            _push(spikes, "tcp_syn_no_ack", t_ms)
            expired.append(k)
    for k in expired:
        st["syn_track"].pop(k, None)
        st["handshake_state"].pop(k, None)

    if flg & FIN:
        st["fin_sent"].add(four)

    if flg & RST and four in st["fin_sent"]:
        _push(spikes, "tcp_fin_to_rst", t_ms)
        st["fin_sent"].remove(four)



# извлечение бинарных признаков UDP
def _handle_udp(
        buf,        # payload IP-пакета, начиная с UDP заголовка
        t_ms,       # время прихода пакета в миллисекундах
        spikes,     # список спайков
        st,         # словарь состояния
        src         # ip источника
    ):
    res = _parse_udp(buf)
    if not res: return
    sport, dport, ulen = res

    # ttfs-кодирование длины udp-сегмента
    bin_ul = _pop_linear(ulen, LEN_BINS)
    norm_ul = _ttfs_norm_linear(ulen, LEN_BINS, bin_ul)
    _push(spikes, f"raw_udp_len_{bin_ul}", _ttfs_apply(t_ms, norm_ul))

    # подозрительные порты
    if sport in SUS_UDP_PORTS or dport in SUS_UDP_PORTS:
        _push(spikes, "udp_port_suspicious", t_ms)

    # новый src-порт
    if sport not in st["ip_ports_udp"][src]:
        if st["ip_ports_udp"][src]:
            _push(spikes, "ip_extended_ports_udp", t_ms)
        st["ip_ports_udp"][src].add(sport)


# извлечение бинарных признаков ICMP
def _handle_icmp(
        dmac,       # mac назначения ethernet-кадра
        ip_src,     # ip источника
        ip_dst,     # ip назначения
        ip_totlen,  # totlen IPv4
        ip_ihl,     # длина заголовка IPv4
        buf,        # срез байтов с ICMP заголовка
        t_ms,       # время прихода пакета в миллисекундах
        spikes      # список спайков
    ):
    res = _parse_icmp(buf)
    if not res: return
    typ, code, ident, seq = res

    # Echo Request
    if typ == 8:
        _push(spikes, "icmp_echo_req", t_ms)
        if _is_broadcast_mac(dmac):
            _push(spikes, "icmp_echo_to_broadcast", t_ms)
    # Echo Reply
    elif typ == 0:
        _push(spikes, "icmp_echo_reply", t_ms)

    # хост/сеть/порт недостижимы
    elif typ == 3:
        # 0=net, 1=host, 2=protocol, 3=port, others
        if code == 3:
            _push(spikes, "icmp_unreach_port", t_ms)

    # ttfs-кодирование размера полезной нагрузки
    icmp_hdr = 8 if typ in (0, 8) else 4
    plen = max(0, ip_totlen - ip_ihl - icmp_hdr)
    bin_icmp = _pop_linear(plen, ICMP_PL_BINS)
    norm_icmp = _ttfs_norm_linear(plen, ICMP_PL_BINS, bin_icmp)
    _push(spikes, f"raw_icmp_pl_{bin_icmp}", _ttfs_apply(t_ms, norm_icmp))


# главная функция кодирования пакета в список спайков
def encode_packet(ts_s, raw:bytes, st):
    # время прихода пакета в мс
    t_ms = ts_s * 1000.0
    # инициализация списка спайков
    spikes = []

    # парсинг Ethernet (L2)
    eth = _parse_eth(raw)
    if not eth:
        _push(spikes, "generic_pkt", t_ms)
        return spikes
    
    etype, smac, dmac, payload, vlan_id, vlan_depth = eth

    # тип mac-адреса назначения
    if _is_broadcast_mac(dmac):
        _push(spikes, "eth_dst_broadcast", t_ms)
    elif _is_multicast_mac(dmac):
        _push(spikes, "eth_dst_multicast", t_ms)

    # тип mac-адреса источника
    if _is_local_admin_mac(smac):
        _push(spikes, "eth_src_local_admin", t_ms)

    # "спайки" уровней L3/L4
    if etype == 0x0806:
        arp = _parse_arp(payload)
        if not arp:
            _push(spikes, "generic_pkt", t_ms)
            return spikes
        op, sha, spa, tpa = arp

        st["ip2mac"][spa].add(sha)
        if len(st["ip2mac"][spa]) >= 2:
            _push(spikes, "ip_multi_mac", t_ms)

        # who-has (request)
        if op == 1:
            if not _is_broadcast_mac(dmac):
                _push(spikes, "arp_unicast_request", t_ms)
            st["mac2tpa"][sha].add(tpa)
            if len(st["mac2tpa"][sha]) >= ARP_REQ_TH:
                _push(spikes, "arp_many_req", t_ms)
            st["arp_pending"][tpa] = ts_s
            if spa == tpa:
                _push(spikes, "arp_gratuitous", t_ms)

        # is-at (reply)
        elif op == 2:
            pend = st["arp_pending"].pop(spa, None)
            if pend is None or ts_s - pend > 60:
                _push(spikes, "arp_reply_without_request", t_ms)
            if len(st["ip2mac"][spa]) >= 2:
                _push(spikes, "arp_reply_diff_mac", t_ms)

        return spikes
    
    # IPv4
    if etype == 0x0800:
        iv = _parse_ipv4(payload)
        if not iv:
            _push(spikes, "generic_pkt", t_ms)
            return spikes
        ttl, proto, totlen, ident, ip_src, ip_dst, df, mf, frag_off, ihl, ip_pl = iv

        # ttfs-кодирование ttl
        bin_ttl = _pop_linear(ttl, TTL_BINS)
        norm_ttl = _ttfs_norm_linear(ttl, TTL_BINS, bin_ttl)
        _push(spikes, f"raw_ttl_{bin_ttl}", _ttfs_apply(t_ms, norm_ttl))

        # отклонние ttl от базового значения
        base = st["ttl_base"].setdefault(ip_dst, ttl)
        if ttl - base >= INC_TH_TTL:
            _push(spikes, "ttl_increase", t_ms)
        if ttl < base:
            st["ttl_base"][ip_dst] = ttl

        # ttfs-кодирование ip total length
        bin_il = _pop_linear(totlen, LEN_BINS)
        norm_il = _ttfs_norm_linear(totlen, LEN_BINS, bin_il)
        _push(spikes, f"raw_ip_len_{bin_il}", _ttfs_apply(t_ms, norm_il))

        # признаки фрагментации
        if mf or frag_off != 0:
            _push(spikes, "ip_fragmented", t_ms)

            # overlap
            key = (ip_src, ip_dst, ident, proto)
            payload_len = max(0, totlen - ihl)
            start8 = frag_off
            end8 = frag_off + ((payload_len + 7) // 8)
            
            if frag_off == 0 and mf:
                st["frag_last_end"][key] = 0
                _push(spikes, "ip_frag_first", t_ms)
            elif frag_off != 0 and mf:
                _push(spikes, "ip_frag_middle", t_ms)
            elif frag_off != 0 and not mf:
                _push(spikes, "ip_frag_last", t_ms)

            last_end = st["frag_last_end"].get(key, 0)
            if start8 < last_end:
                _push(spikes, "ip_frag_overlap", t_ms)

            if end8 > last_end:
                st["frag_last_end"][key] = end8

            # tiny fragments (payload after IPv4 header)
            if payload_len <= TINY_FRAG_PL_TH:
                _push(spikes, "ip_frag_tiny", t_ms)

            # Close tracking on last fragment
            if (frag_off != 0) and (not mf):
                st["frag_last_end"].pop(key, None)


        # извлечение TCP/UDP/ICMP признаков
        if proto == 6 and frag_off == 0:
            _handle_tcp(ip_src, ip_dst, ip_pl, ts_s, t_ms, spikes, st)
        elif proto == 17 and frag_off == 0:
            _handle_udp(ip_pl, t_ms, spikes, st, ip_src)
        elif proto == 1 and frag_off == 0:
            _handle_icmp(dmac, ip_src, ip_dst, totlen, ihl, ip_pl, t_ms, spikes)
        else:
            pass

        return spikes

    # IPv6
    if etype == 0x86DD:
        ip6 = _parse_ipv6(payload)
        if not ip6:
            _push(spikes, "generic_pkt", t_ms)
            return spikes
        hop, nxt, plen, src6, dst6, pl6 = ip6

        # ttfs-кодирование размера данных
        bin_v6l = _pop_linear(plen, LEN_BINS)
        norm_v6l = _ttfs_norm_linear(plen, LEN_BINS, bin_v6l)
        _push(spikes, f"raw_len_{bin_v6l}", _ttfs_apply(t_ms, norm_v6l))

        # извлечение TCP/UDP признаков
        if nxt == 6:
            _handle_tcp(src6, dst6, pl6, ts_s, t_ms, spikes, st)
        elif nxt == 17:
            _handle_udp(pl6, t_ms, spikes, st, src6)

        return spikes

    # другие ethertype
    _push(spikes, "generic_pkt", t_ms)
    return spikes




