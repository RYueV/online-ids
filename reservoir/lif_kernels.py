"""
Динамика LIF-нейронов + расчет рекуррентного тока
"""
from typing import Dict, Optional
import numpy as np
from config import (
    DT,         # шаг интегрирования
    TAU_M,      # постоянная времени утечки мембранного потенциала
    TAU_SYN_E,  # утечка синаптического тока E-нейронов
    TAU_SYN_I,  # утечка синаптического тока I-нейронов
    TAU_W,      # как долго держится "усталость" (SFA/AHP)
    B_ADAPT,    # прирост "усталости" на спайк (SFA/AHP)
    VRESET,     # потенциал покоя 
    REFR        # длительность рефрактерного периода
)

# коэффициенты затухания
# LAM_E = np.exp(-DT/TAU_SYN_E).astype(np.float32)
# LAM_I = np.exp(-DT/TAU_SYN_I).astype(np.float32)
# LAM_W = np.exp(-DT/TAU_W).astype(np.float32)


###
# квант времени для кольцевого буфера (1 мс)
DELAY_BIN_S = 0.001
###



# экспоненциальное затухание синаптических следов
def decay_traces(hid_state:Dict[str,object], dt:float=DT):
    xE = hid_state.get("xE", None)
    lamE = np.float32(np.exp(-dt/TAU_SYN_E))
    if isinstance(xE, np.ndarray):
        np.multiply(xE, lamE, out=xE)

    xI = hid_state.get("xI", None)
    lamI = np.float32(np.exp(-dt/TAU_SYN_I))
    if isinstance(xI, np.ndarray):
        np.multiply(xI, lamI, out=xI)


# рекуррентный ток через синаптические следы и задержки
def recurrent_current_with_delays(
        # состояние скрытого слоя сети
        hid_state:Dict[str,object],
        # словарь ребер и весов
        conn:Dict[str,object],
        use_csr_if_available:bool=False,
        dt:float=DT
    ):
    spikes_prev = hid_state["spikes"]
    N = int(spikes_prev.size)
    
    _ensure_delay_runtime(hid_state, conn)

    if "by_delay" not in conn:
        I_rec = np.zeros(N, dtype=np.float32)
        if use_csr_if_available and isinstance(conn.get("csr", None), dict) and "indptr" in conn["csr"]:
            csr = conn["csr"]
            _csr_matvec_add(csr["indptr"], csr["indices"], csr["data"], spikes_prev, I_rec)
        else:
            _edge_matvec_add(conn["src"], conn["dst"], conn["w"], spikes_prev, I_rec)
        return I_rec

    by_delay = conn["by_delay"]
    L = int(conn["max_delay"]) + 1
    buf = hid_state["delay_buf"]
    head = int(hid_state.get("delay_head", 0))
    xE = hid_state["xE"]
    xI = hid_state["xI"]

    decay_traces(hid_state, dt)

    for d_ms, bloc in by_delay.items():
        if d_ms == 0:
            xbin = spikes_prev
        else:
            idx = (head - int(d_ms)) % L
            xbin = buf[idx]
        if bloc["srcE"].size:
            _edge_matvec_add(bloc["srcE"], bloc["dstE"], bloc["wE"], xbin, xE)
        if bloc["srcI"].size:
            _edge_matvec_add(bloc["srcI"], bloc["dstI"], bloc["wI"], xbin, xI)

    I_rec = xE - xI

    return I_rec


# шаг интеграции LIF + адаптация (SFA/AHP)
def integrate_lif(
        # состояние скрытого слоя сети
        hid_state:Dict[str,object],
        # суммарный входной ток (внешний+рекуррентный)
        I_syn:np.ndarray,
        # текущие значения порогов для каждого нейрона
        vth:np.ndarray,
        # текущие значения tau_m
        tau_m:Optional[np.ndarray]=None,
        # текущее значение шага симуляции
        dt:float=DT
    ):
    # текущие потенциалы нейронов
    V = hid_state["V"]
    # адаптационный ток (sfa/ahp)
    w = hid_state["w"]
    # остаток рефрактерного периода
    ref = hid_state["ref_left"]
    # спайки текущего шага
    spikes = hid_state["spikes"]

    # маска активных нейронов
    active = ref <= 0.0

    # интеграция мембраны только для активных
    if tau_m is None:   # случай постоянной tau_m (скаляр)
        dt_over_tau = np.float32(dt/TAU_M)
        dV = (-V + I_syn - w) * dt_over_tau
        V[active] += dV[active]
    else: # векторная tau_m (изменяющиеся значения)
        # индексы активных нейронов
        idx = np.nonzero(active)[0]
        if idx.size:
            dt_over_tau_loc = (dt/ tau_m[idx]).astype(np.float32, copy=False)
            V[idx] += (-V[idx] + I_syn[idx] - w[idx]) * dt_over_tau_loc
    
    # уменьшение длительности рефрактера на значение шага
    # (для неактивных нейронов)
    ref[~active] -= np.float32(dt)
        
    # проверка порога и формирование спайков
    spikes.fill(0)
    fired = (V >= vth) & active
    spikes[fired] = 1

    # сброс потенциала и рефрактер у fired
    V[fired] = np.float32(VRESET)
    ref[fired] = np.float32(REFR)

    # SFA/AHP
    lamW = np.float32(np.exp(-dt/TAU_W))
    np.multiply(w, lamW, out=w)
    if np.any(fired):
        w[fired] += np.float32(B_ADAPT)



# запись спайков в кольцевой буфер
def delay_buffer_push(
        # состояние скрытого слоя сети
        hid_state:Dict[str,object],
        # словарь ребер и весов
        conn:Dict[str,object],
        # текущий шаг симуляции
        dt:float=DT
    ):
    if "by_delay" not in conn:
        return
    L = int(conn.get("max_delay", 0)) + 1
    if L <= 0:
        L = 1

    steps = int(round(dt / DELAY_BIN_S))
    if steps < 1:
        steps = 1

    buf = hid_state["delay_buf"]
    head = int(hid_state.get("delay_head", 0))
    buf[head, :] = hid_state["spikes"].astype(np.int8, copy=False)
    head = (head + 1) % L

    for _ in range(steps - 1):
        buf[head, :].fill(0)
        head = (head + 1) % L

    hid_state["delay_head"] = head



### группировка ребер по задержке (и типу pre-нейрона)
def _build_by_delay(src, dst, stype, w, delay):
    unique = np.unique(delay.astype(np.int64))
    by_delay = {}
    for d in unique.tolist():
        m_d = (delay == d)
        if not np.any(m_d): continue

        m_E = m_d & ((stype == 0) | (stype == 1))   # preE
        m_I = m_d & ((stype == 2) | (stype == 3))   # preI

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


# инициализация структур для задержек и следов (кольцевой буфер)
def _ensure_delay_runtime(hid_state, conn):
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
###


###
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
###