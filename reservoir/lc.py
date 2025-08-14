"""
locus coeruleus: модуляция порогов и tau_m
"""

from typing import Dict, Optional
import numpy as np
from config import (
    DT,     # шаг интегрирования
    TAU_LC, # утечка "тревоги" 
    A0_LC,  # базовый уровень "тревоги"
    LC_REF, # длительность рефрактерного периода
    K_TH,   # сила воздействия LC на пороги
    K_TAU,  # сила воздействия LC на tau_m (утечку мембранного потенциала скрытых нейронов)
    K_VIP   # сила дезингибиции входов
)


# обновление уровня "тревоги" LC
def lc_update(
        hid_state:Dict[str,object],
        n_alert_spikes:int
    )->None:
    # уровень "тревоги"
    alpha = float(hid_state.get("alpha", 0.0))
    # рефрактер LC
    lc_ref = float(hid_state.get("lc_ref", 0.0))

    # если рефрактер не вышел, входы игнорируются
    if lc_ref > 0.0:
        lc_ref = max(0.0, lc_ref - DT)
        delta_in = 0.0
    else:
        # учет вклада alert-спайков
        delta_in = A0_LC * (1.0 - alpha) * float(max(0, int(n_alert_spikes)))
        if delta_in > 0.0:
            lc_ref = LC_REF
        
    # утечка
    alpha = (alpha + delta_in) * np.exp(-DT/TAU_LC)
    # alpha in [0,1]
    alpha = 0.0 if not np.isfinite(alpha) else float(np.clip(alpha, 0.0, 1.0))

    hid_state["alpha"] = alpha
    hid_state["lc_ref"] = lc_ref


# LC-модуляция
def lc_apply_modulations(
        # состояние скрытого слоя сети
        hid_state:Dict[str,object],
        # разметка нейронов скрытого слоя
        masks:Dict[str,object],
        *,
        # максимальное снижение порога (доля)
        cap_drop:float=0.30,
        # применять модуляцию к tau_m
        use_tau:bool=True,
        # применять дезингибицию входов
        use_gates:bool=False
    )->None:
    # уровень "тревоги"
    alpha = float(hid_state.get("alpha", 0.0))
    # базовые пороги LIF-нейронов
    vth0 = hid_state["vth0"]
    # текущие пороги LIF-нейронов
    vth = hid_state["vth"]
    # уровень "чувствительности" секторов к LC
    sens = masks["sens_of"]

    # если уровень "тревоги" низкий, возвращаются базовые параметры
    if alpha <= 1e-6:
        if vth is not vth0:
            np.copyto(vth, vth0)
        if use_tau and isinstance(hid_state.get("tau_m0", None), np.ndarray):
            tau_m0 = hid_state["tau_m0"]
            tau_m = hid_state["tau_m"]
            if tau_m is not None and tau_m0 is not None and tau_m is not tau_m0:
                np.copyto(tau_m, tau_m0)
        if use_gates and isinstance(hid_state.get("g_inh_base", None), np.ndarray):
            g0 = hid_state["g_inh_base"]
            ge = hid_state["g_inh_eff"]
            if ge is not None and g0 is not None and ge is not g0:
                np.copyto(ge, g0)
        return
    
    # модуляция порогов
    scale_th = 1.0 - (K_TH * alpha) * sens
    min_scale = 1.0 - float(max(0.0, min(1.0, cap_drop)))
    np.maximum(scale_th, min_scale, out=scale_th)
    np.minimum(scale_th, 1.0, out=scale_th)
    np.multiply(vth0, scale_th, out=vth)

    # модуляция tau_m
    if use_tau and isinstance(hid_state.get("tau_m0", None), np.ndarray):
        tau_m0 = hid_state["tau_m0"]
        tau_m = hid_state["tau_m"]
        if tau_m is not None and tau_m0 is not None:
            scale_tau = 1.0 + (K_TAU * alpha) * sens
            np.maximum(scale_tau, 1.0, out=scale_tau)
            np.multiply(tau_m0, scale_tau, out=tau_m)

    # дезингибиция входов
    if use_gates and isinstance(hid_state.get("g_inh_base", None), np.ndarray):
        g0 = hid_state["g_inh_base"]
        ge = hid_state["g_inh_eff"]
        if ge is not None and g0 is not None:
            scale_gate = 1.0 - (K_VIP * alpha) * sens
            np.clip(scale_gate, 0.0, 1.0, out=scale_gate)
            np.multiply(g0, scale_gate, out=ge)


# учет входных сигналов "безопасности"
def lc_apply_safety(
        # состояние скрытого слоя
        hid_state:Dict[str,object],
        # количество безопасных сигналов за шаг
        n_safety_hits:int,
        # сила влияние "безопасных" сигналов
        beta:float=0.10
    )->None:
    if n_safety_hits <= 0: return
    alpha = float(hid_state.get("alpha", 0.0))
    alpha = max(0.0, alpha - float(beta) * float(n_safety_hits))
    hid_state["alpha"] = alpha
