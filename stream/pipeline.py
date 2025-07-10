import dpkt
from encoder.raw2spikes import init_ttfs_state, encode_packet
from lsm.reservoir import init_reservoir_state, reservoir_step
from .utils import SPIKE_QUEUE, pop_ready_spikes
from config import RESERVOIR_PARAMS

PCAP_PATH = ".pcap"


def streaming(pcap_path=PCAP_PATH):
    input_state = init_ttfs_state(dmax_ms=20)
    hidden_params = RESERVOIR_PARAMS
    hidden_state = init_reservoir_state(hidden_params)

    with open(pcap_path, "rb") as f:
        reader = dpkt.pcap.Reader(f)

        for pkt_ts_s, buf in reader:
            pkt_ts_ms = pkt_ts_s * 1000.0

            # advance reservoir until the packet arrives
            while hidden_state['time_ms'] + hidden_params['dt'] <= pkt_ts_ms:
                inputs = pop_ready_spikes(
                    current_time_ms=hidden_state['time_ms']
                )
                reservoir_step(
                    params=hidden_params,
                    state=hidden_state,
                    inputs=inputs
                )
            
            # encode current packet, schedule new spikes
            encode_packet(
                ts_s=pkt_ts_s,
                raw_bytes=buf,
                state=input_state
            )

            # deliver spikes that happen exactly now
            inputs_now = pop_ready_spikes(
                current_time_ms=hidden_state['time_ms']
            )
            if inputs_now:
                reservoir_step(
                    params=hidden_params,
                    state=hidden_state,
                    inputs=inputs_now
                )
        
        # flush remaining spikes after the last packet
        while SPIKE_QUEUE:
            inputs = pop_ready_spikes(
                current_time_ms=hidden_state['time_ms']
            )
            reservoir_step(
                params=hidden_params,
                state=hidden_state,
                inputs=inputs
            )
        
        # TODO: readout




if __name__ == "__main__":
    streaming(PCAP_PATH)




    

