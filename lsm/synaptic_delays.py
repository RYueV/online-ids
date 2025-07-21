import numpy as np

class DelayBuffer:
    def __init__(self, buf_len, max_per_tick=64):
        self.post = np.zeros((buf_len, max_per_tick), dtype=np.int32)
        self.w = np.zeros((buf_len, max_per_tick), dtype=np.float32)
        self.size = np.zeros(buf_len, dtype=np.int16)
        self.max = max_per_tick
        self.len = buf_len

    def schedule(self, tick, post, w):
        pos = tick % self.len
        n = self.size[pos]
        if n >= self.max: return
        self.post[pos, n] = post
        self.w[pos, n] = w
        self.size[pos] += 1

    def deliver(self, tick, i_syn):
        pos = tick % self.len
        n = self.size[pos]
        if n:
            i_syn[self.post[pos, :n]] += self.w[pos, :n]
            self.size[pos] = 0

    def empty(self):
        return not self.size.any()
    

def fast_forward(state, delta_ms):
    if delta_ms <= 0: return
    k = int(round(delta_ms / (state['dt'] + 1e-6)))
    state['v'] *= state['decay_v']**k
    state['i_syn'] = state['decay_i']**k
    state['tick'] += k
    state['time_ms'] += delta_ms


def delay_buffer_empty(state):
    return state['delay'].empty()


def switch_dt(state, params, new_dt):
    old_dt = params['dt']
    params['dt'] = new_dt
    state['decay_v'] = np.exp(-new_dt/params['tau_mem'])
    state['decay_i'] = np.exp(-new_dt/params['tau_syn'])
    state['t_ref_ticks'] = int(np.ceil(params['t_ref']/new_dt))

