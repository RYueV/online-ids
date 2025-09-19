"""
lc -> следы -> рекуррентный ток -> lif
"""
from typing import Dict
import numpy as np

from config import DT
from .lc import lc_update, lc_apply_modulations, lc_apply_safety
from .lif_kernels import (
    recurrent_current_with_delays, 
    delay_buffer_push,
    integrate_lif
)



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
        use_csr_if_available:bool=False,
        # длительность текущего симуляционного шага (сек)
        dt:float=DT
    ):
    # LC: обновить уровень "тревоги", учесть "безопасность", применить модуляцию
    if n_safety_hits > 0:
        lc_apply_safety(hid_state, n_safety_hits)
    lc_update(hid_state, n_alert_spikes)
    lc_apply_modulations(hid_state, masks)

    # спайки за прошлый шаг
    spikes_prev = hid_state["spikes"]
    N = int(spikes_prev.size)   

    # рекуррентный ток через синаптические следы и задержки
    I_rec = recurrent_current_with_delays(hid_state, conn, use_csr_if_available=use_csr_if_available)
    I_syn = I_rec.astype(np.float32, copy=True)

    # внешний ток
    I_ext = hid_state.get("in_curr", 0.0)

    if isinstance(I_rec, np.ndarray):
        if I_ext.size != N:
            raise ValueError(f"размерности массивов тока не совпадают")
        I_syn += I_ext.astype(np.float32)
    else:
        I_syn += float(I_ext)

    # шаг lif
    vth = hid_state["vth"]
    tau_m = hid_state.get("tau_m", None)
    integrate_lif(hid_state, I_syn, vth, tau_m)

    # после интеграции кладем свежие спайки в буфер для будущих задержек
    delay_buffer_push(hid_state, conn)

    if isinstance(I_ext, np.ndarray):
        I_ext.fill(0.0)
    else:
        hid_state["in_curr"] = 0.0
    
    # увеличение симуляционного времени 
    hid_state["t"] += dt
