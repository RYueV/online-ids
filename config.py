###################################################

"""     INPUT LAYER  (КОДЕР ПАКЕТОВ В СПАЙКИ)   """

###################################################
# feature : input_neuron_id
INPUT_MAP = {
    # IPv4 TTL limit
    "raw_ttl_0": 0,
    "raw_ttl_1": 1,
    "raw_ttl_2": 2,
    "raw_ttl_3": 3,
    "raw_ttl_4": 4,
    "raw_ttl_5": 5,

    # Length bins (IP total len)
    "raw_ip_len_0": 6,
    "raw_ip_len_1": 7,
    "raw_ip_len_2": 8,
    "raw_ip_len_3": 9,
    "raw_ip_len_4": 10,
    "raw_ip_len_5": 11,
    "raw_ip_len_6": 12,
    "raw_ip_len_7": 13,

    # TCP window (len(WIN_POWS)+1)
    "raw_win_0": 14,
    "raw_win_1": 15,
    "raw_win_2": 16,
    "raw_win_3": 17,
    "raw_win_4": 18,
    "raw_win_5": 19,
    "raw_win_6": 20,
    "raw_win_7": 21,
    "raw_win_8": 22,

    # dseq (+ special zero/one)
    "raw_dseq_0": 23,   
    "raw_dseq_1": 24,
    "raw_dseq_2": 25,
    "raw_dseq_3": 26,
    "raw_dseq_4": 27,
    "raw_dseq_5": 28,
    "raw_dseq_6": 29,

    # dack (+ special zero/one)
    "raw_dack_0": 30,
    "raw_dack_1": 31,
    "raw_dack_2": 32,
    "raw_dack_3": 33,
    "raw_dack_4": 34,
    "raw_dack_5": 35,
    "raw_dack_6": 36,

    # ICMP payload length bins
    "raw_icmp_pl_0": 37,
    "raw_icmp_pl_1": 38,
    "raw_icmp_pl_2": 39,
    "raw_icmp_pl_3": 40,
    "raw_icmp_pl_4": 41,
    "raw_icmp_pl_5": 42,
    "raw_icmp_pl_6": 43,

    # TCP MSS option bins
    "tcp_opt_mss_0": 44,
    "tcp_opt_mss_1": 45,
    "tcp_opt_mss_2": 46,
    "tcp_opt_mss_3": 47,

    # TCP flags
    "flag_syn": 48,
    "flag_synack": 49,
    "flag_rst": 50,
    "flag_ack_rst": 51,
    "flag_fin": 52,
    "flag_urg": 53,
    "flag_ece": 54,
    "flag_cwr": 55,

    # TCP odd/scan combos
    "tcp_null_flags": 56,
    "tcp_xmas_flags": 57,
    "tcp_syn_fin_both": 58,

    # TCP options presence
    "tcp_opts_present": 59,
    "tcp_opt_wscale": 60,
    "tcp_opt_sack_perm": 61,
    "tcp_opt_timestamp": 62,

    # Generic
    "generic_pkt": 63,

    # TTL anomalies
    "ttl_increase": 64,

    # IPv4 fragmentation features
    "ip_fragmented": 65,
    "ip_frag_first": 66,
    "ip_frag_middle": 67,
    "ip_frag_last": 68,
    "ip_frag_tiny": 69,
    "ip_frag_overlap": 70,

    # ICMP types/codes
    "icmp_echo_req": 71,
    "icmp_echo_to_broadcast": 72,
    "icmp_echo_reply": 73,
    "icmp_unreach_port": 74,

    # Ethernet L2 features
    "eth_dst_broadcast": 75,
    "eth_dst_multicast": 76,
    "eth_src_local_admin": 77,

    # ARP/MitM related
    "ip_multi_mac": 78,
    "arp_many_req": 79,
    "arp_reply_diff_mac": 80,
    "arp_reply_without_request": 81,
    "arp_gratuitous": 82,
    "arp_unicast_request": 83,

    # Per-IP novelty
    "ip_extended_ports_udp": 84,
    "ip_extended_ports_tcp": 85,

    # Suspicious ports
    "udp_port_suspicious": 86,
    "tcp_port_suspicious": 87,

    # Handshake/DoS related
    "tcp_syn_no_ack": 88,
    "tcp_syn_small_win": 89,
    "tcp_unexpected_synack": 90,
    "tcp_fin_to_rst": 91,

    # UDP length bins
    "raw_udp_len_0": 92,
    "raw_udp_len_1": 93,
    "raw_udp_len_2": 94,
    "raw_udp_len_3": 95,
    "raw_udp_len_4": 96,
    "raw_udp_len_5": 97,
    "raw_udp_len_6": 98,
    "raw_udp_len_7": 99,
}
# Количество входов
N_INPUTS = len(INPUT_MAP)



"""     ИНДЕКСЫ ВХОДОВ ПО ГРУППАМ    """
# core - общие, срабатывают на каждом пакете
CORE_FEATURE_IDS = {
    # raw_ttl_*
    0, 1, 2, 3, 4, 5,
    # raw_ip_len_*
    6, 7, 8, 9, 10, 11, 12, 13,
    # raw_udp_len_*
    92, 93, 94, 95, 96, 97, 98, 99
}

# common - характерны для легитимного трафика, но не всегда
COMMON_FEATURE_IDS = {
    # raw_win_*
    14, 15, 16, 17, 18, 19, 20, 21, 22,
    # raw_dseq_*
    23, 24, 25, 26, 27, 28, 29,
    # raw_dack_*
    30, 31, 32, 33, 34, 35, 36,
    # raw_icmp_pl_*
    37, 38, 39, 40, 41, 42, 43,
    # TCP options
    59, 60, 61, 62,         # tcp_opts_present, tcp_opt_wscale, tcp_opt_sack_perm, tcp_opt_timestamp 
    # Нормальные TCP-флаги
    49, 52, 53, 54, 55,     # flag_synack, flag_fin, flag_urg, flag_ece, flag_cwr
    # Базовая фрагментация
    65,                     # ip_fragmented
    66, 67, 68,             # ip_frag_first, ip_frag_middle, ip_frag_last
    # ICMP
    71, 73, 74,             # icmp_echo_req, icmp_echo_reply, icmp_unreach_port
    # Ethernet L2 features
    75, 76, 77,             # eth_dst_broadcast, eth_dst_multicast, eth_src_local_admin
    # generic_pkt
    63,
    # flag_syn
    48        
}

# alert - необычные, подозрительные признаки
ALERT_FEATURE_IDS = {
    64,                 # ttl_increase
    69, 70,             # ip_frag_tiny, ip_frag_overlap
    72,                 # icmp_echo_to_broadcast
    84,                 # ip_extended_ports_udp
    86, 87,             # udp_port_suspicious, tcp_port_suspicious
    91,                 # tcp_fin_to_rst
    50, 51,             # flag_rst, flag_ack_rst
    56                  # tcp_null_flags
}

# threat - атакующие признаки, связаны с определенными техниками
THREAT_ARP_FEATURE_IDS  = {
    78,         # ip_multi_mac
    79,         # arp_many_req
    80,         # arp_reply_diff_mac
    81,         # arp_reply_without_request
    82,         # arp_gratuitous
    83          # arp_unicast_request
}
THREAT_SCAN_FEATURE_IDS = {
    57, 58,             # xmas/syn_fin_both
    44, 45, 46, 47,     # tcp_opt_mss_*
    85                  # ip_extended_ports_tcp
}
THREAT_SYN_FEATURE_IDS  = {
    88,         # tcp_syn_no_ack
    89,         # tcp_syn_small_win
    90          # tcp_unexpected_synack
}

# "сигналы безопасности", глушат активность LC
SAFETY_FEATURE_IDS = set()

# Наименование группы входов -> индексы входов
FEATURE_GROUPS = {
    "core": CORE_FEATURE_IDS,
    "common": COMMON_FEATURE_IDS,
    "alert": ALERT_FEATURE_IDS,
    "arp": THREAT_ARP_FEATURE_IDS,
    "syn": THREAT_SYN_FEATURE_IDS,
    "scan": THREAT_SCAN_FEATURE_IDS,
}



###################################################

"""######     ПОДКЛЮЧЕНИЕ INPUT->HIDDEN    ######"""

###################################################

# Группа входов -> скрытый сектор
FEATURE_TARGET_SECTOR = {
    "core": "core",
    "common":"core",
    "alert": "alert",
    "arp": "arp",
    "syn":"syn",
    "scan": "scan"
}

# Сколько E-нейронов активирует один feature_id этой группы 
INPUT_FANOUT = {
    "core" : 1,
    "common" : 3,
    "alert" : 5,
    "arp" : 6,
    "syn" : 6,
    "scan" : 6
}

# Усиление входа для группы (ток/вес входа)
FEATURE_GAIN = {
    "core": 0.15,       # 1.0 * wE_mean
    "common": 0.15,     # 1.0 * wE_mean
    "alert": 0.22,      # 1.5 * wE_mean
    "arp": 0.08,        # 0.5 * wE_mean
    "syn": 0.08,        # 0.5 * wE_mean
    "scan": 0.08        # 0.5 * wE_mean
}

# Детализация маппинга: детерминированный/псевдослучайный
INPUT_DETERMINISTIC = True
INPUT_RNG_SEED = 123




###################################################

"""     HIDDEN LAYER (LC + МИКРОРЕЗЕРВУАРЫ)    """

###################################################

"""     masks.py    """
# Размер каждого микрорезервуара
SECTOR_SIZES = {
    "core" : 240,
    "alert" : 78,
    "arp" : 42,
    "scan" : 47,
    "syn" : 23
}
# Доля E-нейронов в каждом секторе
EI_RATIO = 0.80
# Чувствительность секторов к LC
S_SENS = {
    "core" : 0.0,
    "alert" : 0.2,
    "arp" : 1.0,
    "scan" : 1.0,
    "syn" : 1.0
}

###################################################

"""     topology.py    """
# вероятности внутрисекторных связей
P_INTRA_BY_SECTOR = {
    "core":     {"EE": 0.088, "EI": 0.132, "IE": 0.110, "II": 0.066},
    "alert":    {"EE": 0.097, "EI": 0.145, "IE": 0.121, "II": 0.073},
    "arp":      {"EE": 0.075, "EI": 0.112, "IE": 0.094, "II": 0.056},
    "scan":     {"EE": 0.075, "EI": 0.112, "IE": 0.094, "II": 0.056},
    "syn":      {"EE": 0.075, "EI": 0.112, "IE": 0.094, "II": 0.056},
}
# вероятности связей между секторами
P_INTER_BY_PAIR = {
    ("alert", "arp"):   {"EE": 0.075, "EI": 0.150, "IE": 0.125, "II": 0.0625},
    ("core", "arp"):    {"EE": 0.053, "EI": 0.105, "IE": 0.088, "II": 0.0440},
    ("alert", "scan"):  {"EE": 0.075, "EI": 0.150, "IE": 0.125, "II": 0.0625},
    ("core", "scan"):   {"EE": 0.053, "EI": 0.105, "IE": 0.088, "II": 0.0440},
    ("alert", "syn"):   {"EE": 0.075, "EI": 0.150, "IE": 0.125, "II": 0.0625},
    ("core", "syn"):    {"EE": 0.053, "EI": 0.105, "IE": 0.088, "II": 0.0440},
}

###################################################

"""     динамика LIF    """
# константы времени в секундах

# шаг интеграции (сек)
DT = 0.001
# потенциал покоя
VRESET = 0.0
# порог потенциала LIF-нейронов
VTH = 0.45
# утечка мембранного потенциала
TAU_M = 0.036
# утечка синаптического тока (E/I нейроны)
TAU_SYN_E = 0.018
TAU_SYN_I = 0.010
# SFA/AHP
TAU_W = 0.200
B_ADAPT = 0.10
# длительность рефрактерного периода
REFR = 0.003

###################################################

"""     динамика LC    """
# утечка LC
TAU_LC = 0.020
# базовый уровень тревоги
A0_LC = 0.25
# рефрактер LC
LC_REF = 0.003
# коэффициенты чувствительности к LC
K_TH = 0.20
K_TAU = 0.10
K_VIP = 0.20


###################################################

"""     OUTPUT LAYER     """

###################################################

# Список имен классов для выходного слоя
READOUT_CLASSES = ["normal", "alert", "arp", "syn", "scan"]