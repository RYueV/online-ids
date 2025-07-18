
"""     input layer     """

# feature : input_neuron_id
INPUT_MAP = {
    # one ip maps to multiple MACs (arp mitm, os scan)
    "ip_multi_mac" : 0,
    # ttl increase for same ip (syn dos, os scan)
    "ttl_increase" : 1,
    # tcp syn+ack response (syn dos) 
    "tcp_synack" : 2,
    # tcp rst without ack (os scan, syn dos)
    "tcp_rst" : 3, 
    # tcp syn without ack (syn dos, os scan)
    "tcp_syn" : 4,
    # large udp packet (syn dos)
    "udp_len_large" : 5,
    # new tcp destination port (os scan)
    "new_tcp_dport" : 6,
    # large ip packet (syn dos)
    "ip_totlen_large" : 7,
    # small ip packet (os scan)
    "ip_totlen_small" : 8,
    # tcp null scan (os scan)
    "tcp_null" : 9,
    # tcp packet with fin+psh flags set, syn not set (os scan)
    "tcp_xmas" : 10,
    # tcp packet with only fin flag and zero payload (os scan)
    "tcp_fin_scan" : 11,
    # device announcing its own ip and mac without request (arp mitm)
    "arp_gratuitous" : 12,
    # one ip address observed with different mac addresses in arp replies (arp mitm)
    "arp_dup_ip" : 13,
    # many arp requests sent from a single mac to multiple target ips (arp mitm)
    "arp_many_req" : 14
}

WEIGHTS = {
    "ip_multi_mac" : 8,
    "ttl_increase" : 5,
    "tcp_synack" : 15,
    "tcp_syn" : 6,
    "udp_len_large" : 1,
    "new_tcp_dport" : 4,
    "ip_totlen_large" : 1,
    "ip_totlen_small" : 2,
    "tcp_rst" : 10,    
    "tcp_null" : 10,
    "tcp_xmas" : 10,
    "tcp_fin_scan" : 10,
    "arp_gratuitous" : 10,
    "arp_dup_ip" : 10,
    "arp_many_req" : 10    
}

N_INPUTS = len(INPUT_MAP)


"""     hidden layer     """
N_HIDDEN_NEURONS = 300

N_SECTORS = 5
# input_neuron_id : hidden_sector_id
SECTORS = {
    # normal
    5 : 0, 7 : 0, 8 : 0,
    1 : 0, 3 : 0, 6 : 0,
    # arp mitm                  
    0 : 1, 
    12 : 1, 13 : 1, 14 : 1, 1 : 1,
    # os scan
    4 : 2, 1 : 2, 8 : 2,
    3 : 2, 9 : 2, 10 : 2, 11 : 2,
    # syn dos
    2 : 3, 6 : 3
    # reserve        
    # 20 hidden neurons 
}

SECTOR_SIZES = [70, 70, 70, 70, 20]

RESERVOIR_PARAMS = {
    'seed' : 42,
    'sparsity' : 0.15,
    'inh_frac' : 0.25,
    'dt' : 0.1,
    'delay_min' : 0.1,
    'delay_max' : 4.0,
    # weights of recurrent synapses
    'w_rec_scale' : 0.5,
    # weights of input synapses
    'w_in_min' : 2.0,
    'w_in_max' : 10.0,
    'tau_mem' : 10.0,    # 10-20
    'tau_syn' : 5.0,     # 5-10
    'tau_stdp' : 40.0,
    'v_th_init' : 0.8,
    'v_reset' : 0.0,
    't_ref' : 0.2
}



"""     output layer     """