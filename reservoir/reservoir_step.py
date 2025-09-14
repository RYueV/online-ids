"""
lc -> следы -> рекуррентный ток -> lif
"""
from typing import Dict
import numpy as np

from .lc import lc_update, lc_apply_modulations, lc_apply_safety
from .lif_kernels import decay_traces, integrate_lif


def reservoir_step(
        # состояние скрытого слоя
        hid_state:Dict[str,object],
        # словарь ребер и весов
        conn:Dict[str,object],
        # разметка нейронов скрытого слоя
        masks:Dict[str,object],
        *,
        # количество alert-спайков за шаг
        n_alert_spikes:int,
        # количество активаций "базопасных" признаков за шаг
        n_safety_hits:int=0,
        use_csr_if_available:bool=False  
    )->None:
    # LC: обновить уровень "тревоги", учесть "безопасность", применить модуляцию
    if n_safety_hits > 0:
        lc_apply_safety(hid_state, n_safety_hits)
    lc_update(hid_state, n_alert_spikes)
    lc_apply_modulations(hid_state, masks)

    # синаптические следы
    decay_traces(hid_state)

    # рекуррентный ток
    spikes_prev = hid_state["spikes"]
    N = int(spikes_prev.size)
    I_rec = np.zeros(N, dtype=np.float32)

    csr = conn.get("csr", None) if use_csr_if_available else None
    if isinstance(csr, dict) and "indptr" in csr:
        _csr_matvec_add(csr["indptr"], csr["indices"], csr["data"], spikes_prev, I_rec)
    else:
        src = conn["src"]
        dst = conn["dst"]
        w = conn["w"]
        _edge_matvec_add(src, dst, w, spikes_prev, I_rec)

    # +внешний ток
    in_curr = hid_state["in_curr"]
    I_syn = I_rec
    if in_curr is not None:
        I_syn = I_syn + in_curr
        in_curr.fill(0.0)

    # шаг lif
    vth = hid_state["vth"]
    tau_m = hid_state.get("tau_m", None)
    integrate_lif(hid_state, I_syn, vth, tau_m)



def _csr_matvec_add(indptr, indices, data, x_bin, out):
    n_rows = indptr.size - 1
    for u in range(n_rows):
        xu = x_bin[u]
        if xu == 0: continue
        start = indptr[u]
        end = indptr[u+1]
        if start == end: continue
        if xu == 1:
            out[indices[start:end]] += data[start:end]
        else:
            out[indices[start:end]] += data[start:end] * float(xu)


def _edge_matvec_add(src, dst, w, x_bin, out):
    active = x_bin[src] != 0
    if not np.any(active): return
    src_act = src[active]
    dst_act = dst[active]
    w_act = w[active].astype(np.float32, copy=False)
    xb = x_bin[src_act].astype(np.float32, copy=False)
    if np.all(xb==1.0):
        contrib = w_act
    else:
        contrib = w_act * xb

    np.add.at(out, dst_act, contrib)