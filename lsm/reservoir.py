import numpy as np
from config import (
    N_INPUTS,
    N_HIDDEN_NEURONS,
    SECTOR_SIZE,
    GROUPS
)


def build_input_connections(params, rng):
    # input_from[i] = [target_neuron_id, synaptic_weight, synaptic_delay_in_ticks]
    input_from = [[] for _ in range(N_INPUTS)]
    # for each input channel connect to 4 neurons in its assigned sector + 1 outside
    for in_id in range(N_INPUTS):
        # get sector id based on input type
        try:
            sector_id = GROUPS.get(in_id)
        except:
            sector_id = 4
        first_id = sector_id * SECTOR_SIZE
        last_id = first_id + SECTOR_SIZE if sector_id < 4 else N_HIDDEN_NEURONS
        local_pool = np.arange(first_id, last_id)

        # 4 targets inside the sector (local)
        inside = rng.choice(local_pool, size=min(4, local_pool.size), replace=False)
        # 1 target outside the sector (global)
        outside = rng.choice(
            np.setdiff1d(np.arange(N_HIDDEN_NEURONS), local_pool),
            size=1, replace=False
        )
        # merge local and global targets
        targets = np.concatenate([inside, outside])

        weights = rng.uniform(
            params['w_in_min'],
            params['w_in_max'],
            size=targets.size
        )
        # input spikes have 0 delay 
        syn_delay = np.zeros_like(targets, dtype=int)

        input_from[in_id] = [
            (int(t), float(w), int(d))
            for t, w, d in zip(targets, weights, syn_delay)
        ]

    return input_from



def build_reservoir_graph(params, rng):
    # number of recurrent connections in the reservoir based on desired sparsity
    num_edges = int(
        N_HIDDEN_NEURONS * (N_HIDDEN_NEURONS - 1) * params['sparsity']
    )

    # pre -> post neuron pairs (candidates)
    pre = rng.integers(0, N_HIDDEN_NEURONS, size=num_edges*3, dtype=int)
    post = rng.integers(0, N_HIDDEN_NEURONS, size=num_edges*3, dtype=int)    
    
    # filter self-loops, remove duplicates
    mask = pre != post
    edges = np.unique(
        np.stack((pre[mask], post[mask]), 1),
        axis=0
    )[:num_edges]
    pre, post = edges[:,0], edges[:,1]

    # inhibitory mask and synaptic weights
    is_inh = rng.random(N_HIDDEN_NEURONS) < params['inh_frac']
    weights = rng.exponential(
        params['w_rec_scale'], size=num_edges
    ).astype(np.float32)
    # inhibitory neurons send negative weights
    weights[is_inh[pre]] *= -1.0

    # generate synaptic delays in ms and convert to ticks
    d_ms = rng.uniform(
        params['delay_min'], 
        params['delay_max'], 
        size=num_edges
    )
    d_ticks = np.round(d_ms / params['dt']).astype(int)

    # build adjacency list
    graph_from = [[] for _ in range(N_HIDDEN_NEURONS)]
    for epre, epost, w, dtk in zip(pre, post, weights, d_ticks):
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

    # current position in delay buffer
    pos = tick % buf_len

    # deliver all delayed spikes scheduled for this tick
    for post, w in state['delay_buf'][pos]:
        i_syn[post] += w
    state['delay_buf'][pos].clear()

    # apply input spikes
    for in_id in inputs:
        state['pre_trace'][in_id] = 1.0
        # schedule their effect on connected hidden neurons
        for post, w, dtk in state['input_from'][in_id]:
            buf_pos = (tick + dtk) % buf_len
            state['delay_buf'][buf_pos].append((post, w))

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

    