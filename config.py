
"""     input layer     """
N_INPUTS = 13

# feature : input_neuron_id
INPUT_MAP = {
    "arp_req" : 0,
    "arp_rep" : 1,
    "tcp_syn" : 2,
    "tcp_synack" : 3,
    "tcp_rst" : 4,
    "ssdp" : 5,
    #PACKET_SIZE = (6, 7, 8)
    "other_udp" : 9,
    "other_tcp" : 10,
    "reserve1" : 11,
    "reserve2" : 12
}



"""     hidden layer     """
N_HIDDEN_NEURONS = 300

N_HIDDEN_GROUPS = 5
# input_neuron_id : hidden_sector_id
GROUPS = {
    0 : 0, 1 : 0,                   # ARP
    2 : 1, 3 : 1, 4 : 1,            # TCP
    5 : 2,                          # SSDP
    6 : 3, 7 : 3, 8 : 3,            # packet size
    9 : 4, 10 : 4, 11 : 4, 12 : 4   # other & reserve
}

SECTOR_SIZE = N_HIDDEN_NEURONS // N_HIDDEN_GROUPS

RESERVOIR_PARAMS = {
    'seed' : 42,
    'sparsity' : 0.10,
    'inh_frac' : 0.25,
    'dt' : 0.2,
    'delay_min' : 0.2,
    'delay_max' : 3.0,
    'w_rec_scale' : 18.0,
    'w_in_min' : 10.0,
    'w_in_max' : 40.0,
    'tau_mem' : 20.0,
    'tau_syn' : 6.0,
    'tau_stdp' : 40.0,
    'v_th_init' : 1.0,
    'v_reset' : 0.0,
    't_ref' : 2.0
}



"""     output layer     """