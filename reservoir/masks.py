"""
Разметка нейронов скрытого слоя по секторам и E/I
"""

from typing import Dict, Iterable, Optional
import numpy as np
from config import SECTOR_SIZES, EI_RATIO, S_SENS


def build_indices(
        *,
        # явный порядок секторов
        sector_order:Optional[Iterable[str]]=None,
        # если True, деление E/I детерминировано
        detEI:bool=True,
        # объект ГПСЧ
        rng:Optional[np.random.Generator]=None,
        # постоянная времени утечки мембранного потенциала по секторам (секунды)
        tau_m_by_sector:Optional[Dict[str, float]]=None
    )->Dict[str, object]:
    """
    Возвращает Dict с ключами:
        "idx_sector" - индексы нейронов сектора в глобальной нумерации
        "idxE_sector" - глобальные индексы возбуждающих нейронов внутри каждого сектора
        "idxI_sector" - глобальные индексы тормозных нейронов внутри каждого сектора
        "is_exc" - булев массив, True если нейрон возбуждающий
        "sector_of" - код сектора для каждого нейрона
        "sens_of" - чувствительность нейрона к LC
        "tau_m0" - базовые tau_m по нейронам
    """
    # порядок обхода секторов
    if sector_order is None:
        sector_order = ("core", "alert", "arp", "scan", "syn")
    sector_order = tuple(sector_order)

    # проверка наличия размеров для всех секторов
    for s_name in sector_order:
        if s_name not in SECTOR_SIZES:
            raise ValueError(f"SECTOR_SIZES missing sector '{s_name}'")

    if not (0.0 <= EI_RATIO <= 1.0):
        raise ValueError("EI_RATIO mast be in [0,1]")

    # инициализация ГПСЧ при необходимости
    if (not detEI) and (rng is None):
        rng = np.random.default_rng()

    # список размеров секторов
    sizes = [int(SECTOR_SIZES[s]) for s in sector_order]
    # сдвиги для вычисления глобальных индексов
    offsets = np.cumsum([0]+sizes[:-1])
    # общее количество скрытых нейронов
    N = int(np.sum(sizes, dtype=np.int64))

    # булев массив; True, если E-нейрон
    is_exc = np.zeros(N, dtype=bool)
    # код сектора для каждого нейрона
    sector_of = np.empty(N, dtype=np.int16)
    # чувствительность каждого нейрона к LC
    sens_of = np.empty(N, dtype=np.float32)

    # словари индексов по секторам
    idx_sector:Dict[str, np.ndarray] = {}
    idxE_sector:Dict[str, np.ndarray] = {}
    idxI_sector:Dict[str, np.ndarray] = {}

    # базовые tau_m по нейронам
    tau_m0 = None
    if tau_m_by_sector is not None:
        tau_m0 = np.empty(N, dtype=np.float32)

    # коды секторов
    sec2code = {s_name : s_idx for s_idx, s_name in enumerate(sector_order)}

    # разметка секторов
    for s_name in sector_order:
        # номер сектора
        s_code = sec2code[s_name]
        # смещение индексов для этого сектора
        s_off = offsets[s_code]
        # размер этого сектора
        s_size = sizes[s_code]
        if s_size <= 0:
            raise ValueError(f"Sector '{s_name}' has non-positive size {s_size}")
        
        # глобальные индексы сектора
        gidx = np.arange(s_off, s_off+s_size, dtype=np.int32)
        # {"s_name" : [s_range]}
        idx_sector[s_name] = gidx

        # количество E-нейронов
        nE = int(round(s_size * EI_RATIO))
        nE = max(0, min(nE, s_size))
        # детерменированное разделение E/I
        if detEI:
            e_idx = gidx[:nE]
            i_idx = gidx[nE:]
        # случайное без повторов
        else:
            # формирование отсортированного массива случайных индексов E-нейронов из gidx
            e_idx = np.sort(rng.choice(gidx, size=nE, replace=False)).astype(np.int32)
            # массив размера s_size, заполненный True
            mask = np.ones(s_size, dtype=bool)
            # замена индексов e_idx - s_off (для нумерации с 0) в mask на False
            # т.е. помечаем индексы E-нейронов как зарезервированные
            mask[e_idx - s_off] = False
            # фильтрация: в i_idx попадут только те индексы, которые не заняты в e_idx
            i_idx = gidx[mask]
        
        # глобальные индексы возбуждающих нейронов внутри каждого сектора
        idxE_sector[s_name] = e_idx
        # глобальные индексы тормозных нейронов внутри каждого сектора
        idxI_sector[s_name] = i_idx

        is_exc[e_idx] = True
        sector_of[gidx] = s_code
        sens_of[gidx] = float(S_SENS.get(s_name, 0.0))

        if tau_m0 is not None:
            if s_name in tau_m_by_sector:
                tau_m0[gidx] = float(tau_m_by_sector[s_name])
            else:
                tau_m0[gidx] = np.nan

    return {
        "idx_sector" : idx_sector,
        "idxE_sector" : idxE_sector,
        "idxI_sector" : idxI_sector,
        "is_exc" : is_exc,
        "sector_of" : sector_of,
        "sens_of" : sens_of,
        "tau_m0" : tau_m0,
        "sec2code" : sec2code
    }

