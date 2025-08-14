from typing import Dict
import numpy as np
from config import VTH, VRESET


# инициализация состояния сети
def init_state(
        # разметка нейронов скрытого слоя по секторам
        masks:Dict[str,object],
        *,
        # своя tau_m на каждый сектор
        use_sector_tau_m:bool=True,
        # нужно ли создавать синаптические следы
        create_traces:bool=True,
        # дезингибиция входов в threat-подпулы (аналог VIP->SST)
        create_gates:bool=False,
        g_inh_base_val:float=0.0
    ):
    # глобальные индексы E-нейронов
    is_exc = masks["is_exc"]
    if not isinstance(is_exc, np.ndarray):
        raise TypeError("masks[is_exc] must be a numpy array")
    # количество нейронов в скрытом слое
    N = int(is_exc.size)

    # мембранный потенциал
    v = np.full(N, VRESET, dtype=np.float32)
    # адаптация (SFA/AHP)
    w = np.zeros(N, dtype=np.float32)
    # оставшаяся рефрактерность
    ref_left = np.zeros(N, dtype=np.float32)
    # базовый порог
    vth0 = np.full(N, VTH, dtype=np.float32)
    # текущий порог
    vth = vth0.copy()
    # спайки текущего шага (0/1)
    spikes = np.zeros(N, dtype=np.int8)
    # внешний входной ток на шаг
    in_curr = np.zeros(N, dtype=np.float32)

    # своя постоянная утечки для каждого сектора (если нужно)
    tau_m0 = masks.get("tau_m0", None)
    if use_sector_tau_m and isinstance(tau_m0, np.ndarray):
        # текущие
        tau_m = tau_m0.astype(np.float32, copy=True)
        # базовые
        tau_m0 = tau_m0.astype(np.float32, copy=True)
    else:
        tau_m = None
        tau_m0 = None

    # синаптические следы (если нужно)
    if create_traces:
        xE = np.zeros(N, dtype=np.float32)
        xI = np.zeros(N, dtype=np.float32)
    else:
        xE = None
        xI = None

    # уровень настороженности (для LC)
    alpha = 0.0
    # рефрактер LC, сек
    lc_ref = 0.0

    # дезингибиция (опицонально)
    if create_gates:
        g_inh_base = np.full(N, float(g_inh_base_val), dtype=np.float32)
        g_inh_eff = g_inh_base.copy()
    else:
        g_inh_base = None
        g_inh_eff = None

    state = {
        "t" : 0.0,
        "V" : v,
        "w" : w,
        "ref_left" : ref_left,
        "vth0" : vth0,
        "vth" : vth,
        "tau_m0" : tau_m0,
        "tau_m" : tau_m,
        "spikes" : spikes,
        "in_curr" : in_curr,
        "alpha" : alpha,
        "lc_ref" : lc_ref
    }

    if create_traces:
        state["xE"] = xE
        state["xI"] = xI

    if create_gates:
        state["g_inh_base"] = g_inh_base
        state["g_inh_eff"] = g_inh_eff

    return state
    