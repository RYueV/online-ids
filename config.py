
"""     input layer     """

# feature : input_neuron_id
INPUT_MAP = {
    "ip_multi_mac" : 0,
    "ttl_increase" : 1,
    "tcp_synack" : 2,
    "tcp_rst" : 3, 
    "tcp_syn" : 4,
    "udp_len_large" : 5,
    "new_tcp_dport" : 6,
    "ip_totlen_large" : 7,
    "ip_totlen_small" : 8,
    "tcp_null" : 9,
    "tcp_xmas" : 10,
    "tcp_fin_scan" : 11,
    # these inputs are triggered shortly before the start of the ARP MitM attack
    # (approximately 5 packets before the attack traffic begins) 
    "arp_gratuitous" : 12,
    "arp_dup_ip" : 13,
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

N_HIDDEN_GROUPS = 6
# input_neuron_id : hidden_sector_id
GROUPS = {
    # ARP
    0 : 0, 1 : 0, 23 : 0,
    # TCP                  
    4 : 1, 5 : 1, 6 : 1, 8 : 1, 9 : 1, 10 : 1, 24 : 1, 25 : 1,
    # UDP
    11 : 2, 12 : 2, 13 : 2, 14 : 2, 15 : 2, 26 : 2, 
    # ICMP
    2 : 3, 3 : 3,
    # header/ports/ttl/direction        
    16 : 4, 17 : 4, 18 : 4, 19 : 4, 20 : 4, 21 : 4, 22 : 4,
    # other
    29 : 5
}

SECTOR_SIZE = N_HIDDEN_NEURONS // N_HIDDEN_GROUPS

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