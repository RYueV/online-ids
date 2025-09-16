"""
lc -> следы -> рекуррентный ток -> lif
"""
from typing import Dict
import numpy as np

from .lc import lc_update, lc_apply_modulations, lc_apply_safety
from .lif_kernels import decay_traces, integrate_lif



# один шаг динамики резервуара
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

    # рекуррентный ток через синаптические следы и задержки
    spikes_prev = hid_state["spikes"]
    N = int(spikes_prev.size)

    _ensure_delay_runtime(hid_state, conn, masks)

    if "by_delay" not in conn:
        I_rec = np.zeros(N, dtype=np.float32)
        csr = conn.get("csr", None) if use_csr_if_available else None
        if isinstance(csr, dict) and "indptr" in csr:
            _csr_matvec_add(csr["indptr"], csr["indices"], csr["data"], spikes_prev, I_rec)
        else:
            src = conn["src"]; dst = conn["dst"]; w = conn["w"]
            _edge_matvec_add(src, dst, w, spikes_prev, I_rec)
        pass
    else:
        by_delay = conn["by_delay"]
        L = int(conn["max_delay"]) + 1
        buf = hid_state["delay_buf"]
        head = int(hid_state.get("delay_head", 0))
        xE = hid_state["xE"]
        xI = hid_state["xI"]

        for d, bloc in by_delay.items():
            idx = (head - int(d)) % L
            xbin = buf[idx]
            if bloc["srcE"].size:
                _edge_matvec_add(bloc["srcE"], bloc["dstE"], bloc["wE"], xbin, xE)
            if bloc["srcI"].size:
                _edge_matvec_add(bloc["srcI"], bloc["dstI"], bloc["wI"], xbin, xI)

        # теперь суммарный рекуррентный ток
        I_rec = xE - xI

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

    # после интеграции кладем свежие спайки в буфер для будущих задержек
    if "by_delay" in conn:
        L = int(conn["max_delay"]) + 1
        buf = hid_state["delay_buf"]
        head = int(hid_state.get("delay_head", 0))
        # записываем текущие спайки в head, затем сдвигаем указатель
        buf[head, :] = hid_state["spikes"].astype(np.int8, copy=False)
        hid_state["delay_head"] = (head + 1) % L



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



# группировка ребер по задержке и типу pre-нейрона
def _build_by_delay(src, dst, stype, w, delay):
    unique = np.unique(delay.astype(np.int64))
    by_delay = {}
    for d in unique.tolist():
        m_d = (delay == d)
        if not np.any(m_d): continue

        m_E = m_d & ((stype == 0) | (stype == 1))
        m_I = m_d & ((stype == 2) | (stype == 3))

        bloc = {}
        if np.any(m_E):
            bloc["srcE"] = src[m_E].astype(np.int64, copy=False)
            bloc["dstE"] = dst[m_E].astype(np.int64, copy=False)
            bloc["wE"] = w[m_E].astype(np.float32, copy=False)
        else:
            bloc["srcE"] = np.empty(0, np.int64)
            bloc["dstE"] = np.empty(0, np.int64)
            bloc["wE"] = np.empty(0, np.float32)

        if np.any(m_I):
            bloc["srcI"] = src[m_I].astype(np.int64, copy=False)
            bloc["dstI"] = dst[m_I].astype(np.int64, copy=False)
            bloc["wI"] = (-w[m_I]).astype(np.float32, copy=False)
        else:
            bloc["srcI"] = np.empty(0, np.int64)
            bloc["dstI"] = np.empty(0, np.int64)
            bloc["wI"] = np.empty(0, np.float32)

        by_delay[int(d)] = bloc
    
    max_delay = int(unique.max()) if unique.size > 0 else 0
    return by_delay, max_delay



# инициализация структур для задержек и следов
def _ensure_delay_runtime(hid_state, conn, masks):
    N = int(hid_state["spikes"].size)
    if "xE" not in hid_state or not isinstance(hid_state["xE"], np.ndarray) or hid_state["xE"].size != N:
        hid_state["xE"] = np.zeros(N, dtype=np.float32)
    if "xI" not in hid_state or not isinstance(hid_state["xI"], np.ndarray) or hid_state["xI"].size != N:
        hid_state["xI"] = np.zeros(N, dtype=np.float32)

    if "by_delay" not in conn:
        if ("delay" not in conn) or ("stype" not in conn):
            return
        by_delay, max_delay = _build_by_delay(
            conn["src"].astype(np.int64, copy=False),
            conn["dst"].astype(np.int64, copy=False),
            conn["stype"].astype(np.int64, copy=False),
            conn["w"].astype(np.float32, copy=False),
            conn["delay"].astype(np.int64, copy=False),
        )
        conn["by_delay"] = by_delay
        conn["max_delay"] = max_delay

    L = int(conn.get("max_delay", 0)) + 1
    if L <= 0:
        L = 1
    if ("delay_buf" not in hid_state) or (hid_state["delay_buf"].shape != (L, N)):
        hid_state["delay_buf"] = np.zeros((L, N), dtype=np.int8)
        hid_state["delay_head"] = 0