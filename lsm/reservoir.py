import numpy as np
from collections import defaultdict
from config import (
    N_INPUTS,
    N_HIDDEN_NEURONS,
    SECTOR_SIZES,
    SECTORS,
    WEIGHTS,
    N_SECTORS
)

SECTOR_BOUND = np.cumsum([0] + SECTOR_SIZES)
assert SECTOR_BOUND[-1] == N_HIDDEN_NEURONS, "N_HIDDEN_NEURONS != num_sec_neurons*num_sec"


def sector_of(neuron_id):
    return int(np.searchsorted(SECTOR_BOUND, neuron_id, side='right') - 1)


def neurons_in_sector(sector_id):
    start = SECTOR_BOUND[sector_id]
    stop = SECTOR_BOUND[sector_id + 1]
    return np.arange(start, stop, dtype=int)


def allocate_neuron_quota():
    inputs_by_sector = defaultdict(list)
    for input_id in range(N_INPUTS):
        sector = SECTORS.get(input_id, 4)
        inputs_by_sector[sector].append(input_id)

    quotas = {}

    for sector_id in range(N_SECTORS):
        sec_size = SECTOR_SIZES[sector_id]
        in_list = inputs_by_sector.get(sector_id, [])
        if not in_list: continue

        w_list = np.array([WEIGHTS[i] for i in in_list], dtype=float)
        w_sum = w_list.sum()
        frac = w_list / w_sum

        raw_q = np.maximum(1, np.round(frac * sec_size)).astype(int)
        diff = sec_size - raw_q.sum()

        if diff > 0:
            for _ in range(diff):
                idx = np.argmax(frac - raw_q / sec_size)
                raw_q[idx] += 1
        elif diff < 0:
            for _ in range(-diff):
                idx = np.argmax(raw_q)
                if raw_q[idx] > 1:
                    raw_q -= 1
        
        quotas.update({
            in_id: int(q)
            for in_id, q in zip(in_list, raw_q)
        })
    
    return quotas


def build_input_connections(params, rng):
    # input_from[i] = [target_neuron_id, synaptic_weight, synaptic_delay_in_ticks]
    input_from = [[] for _ in range(N_INPUTS)]

    quota = allocate_neuron_quota()
    in_weights = params.get(
        'w_in_base',
        (params['w_in_min'] + params['w_in_max']) / 2
    )
    all_neurons = np.arange(N_HIDDEN_NEURONS, dtype=int)

    for in_id in range(N_INPUTS):
        n_targets = max(1, quota.get(in_id, 1))
        sector_id = SECTORS.get(in_id, 4)

        local_pool = neurons_in_sector(sector_id)
        global_pool = np.setdiff1d(all_neurons, local_pool, assume_unique=True)

        n_local = min(len(local_pool), max(1, int(0.8 * n_targets)))
        n_global = n_targets - n_local

        local_tg = rng.choice(local_pool, size=n_local, replace=False)
        global_tg = rng.choice(global_pool, size=n_global, replace=False)

        targets = np.concatenate([local_tg, global_tg])
        for t in targets:
            input_from[in_id].append((int(t), float(in_weights), 0))

    return input_from



def build_reservoir_graph(params, rng):
    # number of recurrent connections in the reservoir based on desired sparsity
    num_edges = int(
        N_HIDDEN_NEURONS * (N_HIDDEN_NEURONS - 1) * params['sparsity']
    )
    edges = []

    # pre -> post neuron pairs (candidates)
    while len(edges) < num_edges:
        pre = rng.integers(0, N_HIDDEN_NEURONS)
        post = rng.integers(0, N_HIDDEN_NEURONS)
        # filter self-loops
        if pre == post: continue

        same = sector_of(pre) == sector_of(post)
        if same and rng.random() > 0.7: continue
        
        edges.append((pre, post))

    edges = np.array(edges, dtype=int)
    pre, post = edges[:,0], edges[:,1]

    # inhibitory mask and synaptic weights
    is_inh = rng.random(N_HIDDEN_NEURONS) < params['inh_frac']
    rec_weights = np.abs(rng.normal(
        loc=0.0,
        scale=1.0,
        size=num_edges
    )).astype(np.float32)
    # inhibitory neurons send negative weights
    rec_weights[is_inh[pre]] *= -1.0

    # target std
    current_std = np.std(rec_weights)
    if current_std > 0:
        rec_weights = rec_weights / current_std * params['w_rec_scale']

    # generate synaptic delays in ms and convert to ticks
    d_ms = rng.uniform(
        params['delay_min'], 
        params['delay_max'], 
        size=num_edges
    )
    d_ticks = np.round(d_ms / params['dt']).astype(int)

    # build adjacency list
    graph_from = [[] for _ in range(N_HIDDEN_NEURONS)]
    for epre, epost, w, dtk in zip(pre, post, rec_weights, d_ticks):
        graph_from[epre].append((int(epost), float(w), int(dtk)))
    
    return graph_from



def init_reservoir_state(params):
    rng = np.random.default_rng(params['seed'])
    graph_from = build_reservoir_graph(params, rng)
    input_from = build_input_connections(params, rng)
    dt = params['dt']
    buf_len = int(np.ceil(params['delay_max'] / dt)) + 1

    state = {
        # membrane potentials
        'v' : np.zeros(N_HIDDEN_NEURONS, np.float32),
        # synaptic currents
        'i_syn' : np.zeros(N_HIDDEN_NEURONS, np.float32),
        # firing threshold
        'v_th' : np.full(N_HIDDEN_NEURONS, params['v_th_init'], np.float32),
        # refractory timers
        'ref_until' : np.zeros(N_HIDDEN_NEURONS, int),
        # presynaptic traces (for stdp)
        'pre_trace' : np.zeros(N_INPUTS, np.float32),
        # postsynaptic traces (for stdp)
        'post_trace' : np.zeros(N_HIDDEN_NEURONS, np.float32),
        # ring buffer for delayed spikes
        'delay_buf' : [[] for _ in range(buf_len)],
        # spike counts (for homeostasis)
        'spike_counter' : np.zeros(N_HIDDEN_NEURONS, int),
        # current simulation step
        'tick' : 0,
        # current time in milliseconds
        'time_ms' : 0.0,
        # recurrent connections
        'graph_from' : graph_from,
        # input connections
        'input_from' : input_from,
        # delay buffer length
        'buf_len' : buf_len,
        # refractory duration in ticks
        't_ref_ticks' : int(np.ceil(params['t_ref'] / dt)),
        # membrane decay factor
        'decay_v' : np.exp(-dt / params['tau_mem']),
        # synaptic decay factor
        'decay_i' : np.exp(-dt / params['tau_syn']),
        # trace decay factor (for stdp)
        'decay_trace' : np.exp(-dt / params['tau_stdp']),
        # reset value for membrane
        'v_reset' : params['v_reset'],
        # time step size
        'dt' : dt
    }

    return state


def reservoir_step(params, state, inputs):
    v, i_syn = state['v'], state['i_syn']
    tick = state['tick']
    buf_len = state['buf_len']

    # apply input spikes
    for in_id in inputs:
        state['pre_trace'][in_id] = 1.0
        # schedule their effect on connected hidden neurons
        for post, w, dtk in state['input_from'][in_id]:
            buf_pos = (tick + dtk) % buf_len
            state['delay_buf'][buf_pos].append((post, w))

    # current position in delay buffer
    pos = tick % buf_len

    # deliver all delayed spikes scheduled for this tick
    for post, w in state['delay_buf'][pos]:
        i_syn[post] += w
    state['delay_buf'][pos].clear()

    # exponential decay of synaptic current
    i_syn *= state['decay_i']

    # handle refractory neurons
    active = state['ref_until'] <= tick
    # refractory neurons can't integrate or spike
    i_syn[~active] = 0.0

    # update membrane potential for active LIF-neurons
    v[active] = (
        v[active] * state['decay_v'] +
        (1.0 - state['decay_v']) * i_syn[active]
    )
    # reset membrane potential for refractory neurons
    v[~active] = state['v_reset']

    # detect spikes
    fired = np.where((v >= state['v_th']) & active)[0]
    if fired.size:
        v[fired] = state['v_reset']
        state['ref_until'][fired] = tick+ state['t_ref_ticks']
        i_syn[fired] = 0.0
        state['post_trace'][fired] = 1.0
        state['spike_counter'][fired] += 1
        # schedule spikes to connected neurons with delays
        for neuron in fired.astype(int):
            for post, w, dtk in state['graph_from'][neuron]:
                state['delay_buf'][(tick + dtk) % buf_len].append((post, w))

    # decay stdp-traces
    state['pre_trace'] *= state['decay_trace']
    state['post_trace'] *= state['decay_trace']

    # next simulation step
    state['tick'] += 1
    state['time_ms'] += state['dt']

    return fired.tolist()

    