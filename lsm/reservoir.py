import numpy as np
from collections import defaultdict
from pathlib import Path
from config import (
    N_INPUTS,
    N_HIDDEN_NEURONS,
    SECTOR_SIZES,
    SECTORS,
    WEIGHTS,
    N_SECTORS,
    RESERVOIR_PARAMS
)
from .synaptic_delays import DelayBuffer

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
                    raw_q[idx] -= 1
        
        quotas.update({
            in_id: int(q)
            for in_id, q in zip(in_list, raw_q)
        })
    
    print(quotas)
    return quotas


def build_input_connections(params, rng):
    # input_from[i] = [target_neuron_id, synaptic_weight, synaptic_delay_in_ticks]
    input_from = [[] for _ in range(N_INPUTS)]

    quota = allocate_neuron_quota()
    in_weights = params['w_in_base']
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

    return np.array(input_from, dtype=object)



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



def _cached_inputs(seed, build_conn):
    cache = Path(f"lsm/reservoir_conn_{seed}.npz")
    if cache.exists():
        return np.load(cache, allow_pickle=True)["arr_0"]
    arr = build_conn()
    np.savez_compressed(cache, arr_0=arr)
    return arr


def init_reservoir_state(params=RESERVOIR_PARAMS):
    rng = np.random.default_rng(params['seed'])
    graph_from = build_reservoir_graph(params, rng)
    input_from = _cached_inputs(params["seed"], lambda: build_input_connections(params, rng))
    dt = params['dt']
    buf_len = int(np.ceil(params['delay_max'] / dt)) + 1
    delay = DelayBuffer(buf_len)

    state = {
        # membrane potentials
        'v' : np.zeros(N_HIDDEN_NEURONS, np.float32),
        # synaptic currents
        'i_syn' : np.zeros(N_HIDDEN_NEURONS, np.float32),
        # firing threshold
        'v_th' : np.full(N_HIDDEN_NEURONS, params['v_th_init'], np.float32),
        # refractory timers
        'ref_until' : np.zeros(N_HIDDEN_NEURONS, int),
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
        'decay_v' : np.float32(np.exp(-dt / params['tau_mem'])),
        # synaptic decay factor
        'decay_i' : np.float32(np.exp(-dt / params['tau_syn'])),
        # reset value for membrane
        'v_reset' : params['v_reset'],
        # time step size
        'delay' : delay,
        'dt' : dt
    }

    return state


def reservoir_step(params, state, inputs):
    t = state['tick']
    delay_buf = state['delay']

    # apply input spikes
    for in_id in inputs:
        # schedule their effect on connected hidden neurons
        for post, w, dtk in state['input_from'][in_id]:
            delay_buf.schedule(t + dtk, post, w)

    # deliver all delayed spikes scheduled for this tick
    delay_buf.deliver(t, state['i_syn'])

    v = state["v"]
    i_syn = state["i_syn"]
    v_th = state["v_th"]
    decay_v = state["decay_v"]
    decay_i = state["decay_i"]
    v_reset = state["v_reset"]


    # handle refractory neurons
    # refractory neurons can't integrate or spike
    refractory = state['ref_until'] > t
    active = ~refractory

    # update membrane potential for active LIF-neurons
    i_syn[active] *= decay_i
    v[active] = v[active] * decay_v + (1.0 - decay_v) * i_syn[active]

    # reset membrane potential for refractory neurons
    i_syn[refractory] = 0.0
    v[refractory] = v_reset


    # detect spikes
    fired = np.where(v >= v_th)[0].astype(np.int32)
    if fired.size > 0:
        v[fired] = v_reset
        state["ref_until"][fired] = t + state["t_ref_ticks"]
        state["spike_counter"][fired] += 1
        # schedule spikes to connected neurons with delays
        for neuron in fired:
            for post, w, dtk in state["graph_from"][neuron]:
                delay_buf.schedule(t + dtk, post, w)

    # next simulation step
    state['tick'] += 1
    state['time_ms'] += state['dt']

    return fired

    