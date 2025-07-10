from collections import deque

# global input spike bus; each element is (input_neuron_id, spike_ts)
SPIKE_QUEUE = deque()
EPS = 1e-9


# append a spike into the global queue
def inputFire(input_neuron_id, spike_ts):
    SPIKE_QUEUE.append((input_neuron_id, spike_ts))


# retrieve all spikes accumulated in queue, sort them by time
# and clear the queue
def flush_sorted_spikes():
    events = list(SPIKE_QUEUE)
    SPIKE_QUEUE.clear()
    events.sort(key=lambda x: x[1])
    return events


# extract spikes whose timestamp is'n greater than current_time_ms
def pop_ready_spikes(current_time_ms):
    ready = []
    keep = []
    while SPIKE_QUEUE:
        neur_id, ts = SPIKE_QUEUE.popleft()
        if ts <= current_time_ms + EPS:
            ready.append(neur_id)
        else:
            keep.append((neur_id, ts))
    SPIKE_QUEUE.extendleft(reversed(keep))
    return ready
