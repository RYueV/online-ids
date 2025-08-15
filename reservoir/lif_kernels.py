"""
Динамика LIF-нейронов
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
LAM_E = np.exp(-DT/TAU_SYN_E).astype(np.float32)
LAM_I = np.exp(-DT/TAU_SYN_I).astype(np.float32)
LAM_W = np.exp(-DT/TAU_W).astype(np.float32)

# экспоненциальное затухание синаптических следов
def decay_traces(hid_state:Dict[str,object])->None:
    xE = hid_state.get("xE", None)
    if isinstance(xE, np.ndarray):
        np.multiply(xE, LAM_E, out=xE)
    xI = hid_state.get("xI", None)
    if isinstance(xI, np.ndarray):
        np.multiply(xI, LAM_I, out=xI)


# шаг интеграции LIF + адаптация (SFA/AHP)
def integrate_lif(
        # состояние скрытого слоя сети
        hid_state:Dict[str,object],
        # суммарный входной ток (внешний+рекуррентный)
        I_syn:np.ndarray,
        # текущие значения порогов для каждого нейрона
        vth:np.ndarray,
        # текущие значения tau_m
        tau_m:Optional[np.ndarray]=None
    )->None:
    # текущие потенциалы нейронов
    V = hid_state["V"]
    # веса связей
    w = hid_state["w"]
    # остаток рефрактерного периода
    ref = hid_state["ref_left"]
    # спайки текущего шага
    spikes = hid_state["spikes"]

    # маска активных нейронов
    active = ref <= 0.0

    # интеграция мембраны только для активных
    if tau_m is None:   # случай постоянной tau_m (скаляр)
        dt_over_tau = np.float32(DT/TAU_M)
        dV = (-V + I_syn - w) * dt_over_tau
        V[active] += dV[active]
    else: # векторная tau_m (изменяющиеся значения)
        # индексы активных нейронов
        idx = np.nonzero(active)[0]
        if idx.size:
            dt_over_tau_loc = (DT/ tau_m[idx]).astype(np.float32, copy=False)
            V[idx] += (-V[idx] + I_syn[idx] - w[idx]) * dt_over_tau_loc
    
    # уменьшение длительности рефрактера на значение шага
    # (для неактивных нейронов)
    ref[~active] -= np.float32(DT)
        
    # проверка порога и формирование спайков
    spikes.fill(0)
    fired = (V >= vth) & active
    spikes[fired] = 1

    # сброс потенциала и рефрактер у fired
    V[fired] = np.float32(VRESET)
    ref[fired] = np.float32(REFR)

    # SFA/AHP
    np.multiply(w, LAM_W, out=w)
    if np.any(fired):
        w[fired] += np.float32(B_ADAPT)