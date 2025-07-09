from collections import deque

# global input spike bus; each element is (input_neuron_id, spike_ts)
SPIKE_QUEUE = deque()

def inputFire(input_neuron_id, spike_ts):
    SPIKE_QUEUE.append((input_neuron_id, spike_ts))