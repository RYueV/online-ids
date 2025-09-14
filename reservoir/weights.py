"""
Генерация значений весов для созданных ребер
"""

from typing import Dict, Optional, Tuple
import numpy as np


# коды типов связей
STYPE = {
    "EE" : np.int8(0), "EI" : np.int8(1),
    "IE" : np.int8(2), "II" : np.int8(3)
}


# half-normal по заданному среднему модуля
def _sample_halfnormal_from_mean(
        rng:np.random.Generator,
        n:int,
        mean_abs:float
    ): 
    if n <= 0:
        return np.empty(0, dtype=np.float32)
    sigma = float(mean_abs) * np.sqrt(np.pi/2.0)
    x = rng.normal(loc=0.0, scale=max(1e-7, sigma), size=n).astype(np.float32, copy=False)
    return np.abs(x).astype(np.float32, copy=False)


# положительные веса (нормальное распределение)
def _sample_positive(
        # объект ГПСЧ
        rng:np.random.Generator,
        # необходимое количество величин
        n:int,
        # среднее
        mean:float,
        # дисперсия
        std:float
    ):
    if n <= 0:
        return np.empty(0, dtype=np.float32)
    x = rng.normal(
        loc=mean,
        scale=max(1e-6, std),
        size=n
    ).astype(np.float32)
    x = np.clip(x, a_min=mean*0.1, a_max=None, dtype=np.float32)
    return x


# отображение глобального индекса в локальный в E-пространстве
def _build_index_map(
        glob_ids:np.ndarray
    )->np.ndarray:
    if glob_ids.size == 0:
        return np.full(0, -1, dtype=np.int32)
    max_id = int(glob_ids.max())
    mapper = np.full(max_id+1, -1, dtype=np.int32)
    mapper[glob_ids] = np.arange(glob_ids.size, dtype=np.int32)
    return mapper


# оценка спектрального радиуса матрицы
def estimate_ee_radius(
        # количество E-нейронов
        nE:int,
        # локальные индексы источников (E)
        srcE:np.ndarray,
        # локальные индексы получателей (E)
        dstE:np.ndarray,
        # веса EE-ребер
        wE:np.ndarray,
        *,
        iters:int=20,
        rng:Optional[np.random.Generator]=None
    )->float:
    if nE <= 0 or srcE.size == 0:
        return 0.0
    
    if rng is None:
        rng = np.random.default_rng()
    x = rng.random(nE, dtype=np.float32)
    x /= (np.linalg.norm(x) + 1e-12)

    stop = max(1, iters)
    for _ in range(stop):
        y = np.zeros_like(x)
        np.add.at(y, dstE, wE * x[srcE])
        norm = np.linalg.norm(y)
        if norm <= 1e-12 or not np.isfinite(norm):
            return 0.0
        x = y/norm
        
    y = np.zeros_like(x)
    np.add.at(y, dstE, wE * x[srcE])
    num = float(np.dot(x, y))
    den = float(np.dot(x, x)) + 1e-12
    val = num/den
    return float(abs(val))


# ограничение суммы модулей входящих весов на каждый post нейрон
def _cap_post_incoming_L1(
        dst:np.ndarray,
        w:np.ndarray,
        cap:float
    )->None:
    M = int(w.size)
    if M==0: return

    order = np.argsort(dst, kind="mergesort")
    dst_sorted = dst[order]
    w_sorted = w[order]

    unique_dst, start_idx = np.unique(dst_sorted, return_index=True)
    start_idx = start_idx.astype(np.int64)

    end_idx = np.empty_like(start_idx)
    end_idx[:-1] = start_idx[1:]
    end_idx[-1] = M

    for s, e in zip(start_idx, end_idx):
        block = slice(s, e)
        total = float(np.sum(np.abs(w_sorted[block])))
        if total > cap and total > 0.0:
            scale = cap / total
            w_sorted[block] *= scale

    w[order] = w_sorted


# среднее по массиву
def _mean_arr(arr):
    if arr is None: return 0.0
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0: return 0.0
    finite = np.isfinite(a)
    if not np.any(finite): return 0.0
    return float(a[finite].mean())



# построение csr-структуры по источнику (rows=src)
def build_csr_by_src(
    src:np.ndarray,
    dst:np.ndarray,
    w:np.ndarray,
    *,
    n_total:int,
    )->Dict[str, np.ndarray]:
    M = int(w.size)
    if M == 0:
        return {
            "indptr": np.zeros(n_total + 1, dtype=np.int32),
            "indices": np.empty(0, dtype=np.int32),
            "data": np.empty(0, dtype=np.float32),
        }

    order = np.argsort(src, kind="mergesort")
    src_sorted = src[order]
    indices = dst[order].astype(np.int32, copy=False)
    data = w[order].astype(np.float32, copy=False)

    counts = np.bincount(src_sorted, minlength=n_total).astype(np.int32, copy=False)
    indptr = np.empty(n_total + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    return {"indptr": indptr, "indices": indices, "data": data}


# главная функция генерации весов
def assign_weights(
        # нейроны-источники
        src:np.ndarray,
        # нейроны-получатели,
        dst:np.ndarray,
        # типы связей
        stype:np.ndarray,
        # разметка нейронов по секторам
        masks:Dict[str, object],
        *,
        # объект ГПСЧ
        rng:Optional[np.random.Generator]=None,
        # среднее и дисперсия весов E-нейронов
        e_mean:float=0.15, e_std:float=0.02,
        # среднее и дисперсия весов I-нейронов
        i_mean:float=0.33, i_std:float=0.03,
        # целевой радиус EE-подматрицы
        ee_radius_target:float=0.90,
        ee_power_iters:int=20,
        i_to_e_ratio:Optional[float]=None,
        # лимит L1-нормы входящих весов на post нейрон
        post_abs_max:Optional[float]=None,
        # задержки в целых шагах
        delay_range:Optional[Tuple[int,int]]=None,
        # строить csr по источнику
        build_csr:bool=True
    )->Dict[str,object]:
    """
    Возвращает словарь:
        "w": float32[M] - веса ребер в том же порядке что у src/dst
        "delay": Optional[int16[M]] - задержки в синапсах
        "csr":Optional[Dict] - csr по источнику
        "by_type":Dict - индексы ребер по типам
    """
    if rng is None:
        rng = np.random.default_rng()

    # количество ребер
    M = int(src.size)
    # инициализация вектора весов
    w = np.zeros(M, dtype=np.float32)

    # индексы по типам ребер
    idxEE = np.where(stype == STYPE["EE"])[0]
    idxEI = np.where(stype == STYPE["EI"])[0]
    idxIE = np.where(stype == STYPE["IE"])[0]
    idxII = np.where(stype == STYPE["II"])[0]

    # ребра, исходящие из E, положительны
    if idxEE.size:
        # w[idxEE] = _sample_positive(rng, idxEE.size, e_mean, e_std)
        w[idxEE] = _sample_halfnormal_from_mean(rng, idxEE.size, e_mean)
    if idxEI.size:
        # w[idxEI] = _sample_positive(rng, idxEI.size, e_mean, e_std)
        w[idxEI] = _sample_halfnormal_from_mean(rng, idxEI.size, e_mean)
        
    # тормозные (IE, II) отрицательны
    if idxIE.size:
        # w[idxIE] = -_sample_positive(rng, idxIE.size, i_mean, i_std)
        w[idxIE] = -_sample_halfnormal_from_mean(rng, idxIE.size, i_mean)
    if idxII.size:
        # w[idxII] = -_sample_positive(rng, idxII.size, i_mean, i_std)
        w[idxII] = -_sample_halfnormal_from_mean(rng, idxII.size, i_mean)

    # масштабирование EE до целевого спектрального радиуса
    if idxEE.size:
        is_exc = masks["is_exc"].astype(bool, copy=False)
        # глобальные индексы всех E-нейронов
        e_glob_idx = np.where(is_exc)[0]
        # отображение глобального индекса в локальный в E-пространстве
        e_map = _build_index_map(glob_ids=e_glob_idx)

        # локальеые индексы EE ребер
        ee_src_loc = e_map[src[idxEE]]
        ee_dst_loc = e_map[dst[idxEE]]
        ee_w = w[idxEE].astype(np.float32, copy=False)

        # оценка спектрального радиуса
        radius = estimate_ee_radius(
            nE=e_glob_idx.size,
            srcE=ee_src_loc,
            dstE=ee_dst_loc,
            wE=ee_w,
            iters=ee_power_iters,
            rng=rng
        )
        if radius > 0 and np.isfinite(radius):
            scale = ee_radius_target / float(radius)
            # ограничение масштаба
            scale = float(np.clip(scale, 0.2, 5.0))
            w[idxEE] *= scale

    # выравнивание среднего |wI| относительно |wE|
    if i_to_e_ratio is not None:
        meanE = _mean_arr(np.abs(w[np.concatenate([idxEE, idxEI])])) if (idxEE.size or idxEI.size) else 0.0
        meanI = _mean_arr(np.abs(w[np.concatenate([idxIE, idxII])])) if (idxIE.size or idxII.size) else 0.0
        if meanE > 0 and meanI >= 0:
            targetI = i_to_e_ratio * meanE
            if meanI > 0:
                scaleI = targetI / meanI
                scaleI = float(np.clip(scaleI, 0.5, 2.5))
                if idxIE.size:
                    w[idxIE] *= scaleI
                if idxII.size:
                    w[idxII] *= scaleI
            else:
                if idxIE.size:
                    w[idxIE] = -_sample_positive(rng, idxIE.size, targetI, targetI * 0.5)
                if idxII.size:
                    w[idxII] = -_sample_positive(rng, idxII.size, targetI, targetI * 0.5)
                
    
    def _renorm_block(idxs, target_abs, negative=False):
        if idxs.size == 0:
            return
        cur = float(np.mean(np.abs(w[idxs])))
        if cur > 0:
            scale = float(target_abs / cur)
            w[idxs] *= scale
        if negative:
            # строго отрицательные тормозные веса (на случай численных огрехов)
            w[idxs] = -np.abs(w[idxs])

    _renorm_block(idxEE, e_mean, negative=False)
    _renorm_block(idxEI, e_mean, negative=False)
    _renorm_block(idxIE, i_mean, negative=True)
    _renorm_block(idxII, i_mean, negative=True)
    
    
    if post_abs_max is not None and post_abs_max > 0:
        _cap_post_incoming_L1(dst, w, cap=float(post_abs_max))
    
    delay = None
    if delay_range is not None:
        low, high = int(delay_range[0]), int(delay_range[1])
        if low < 0 or high < low:
            raise ValueError("delay_range must be (low<=high), low>=0")
        delay = rng.integers(low, high + 1, size=M, dtype=np.int16)

    csr = None
    if build_csr:
        csr = build_csr_by_src(src, dst, w, n_total=masks["is_exc"].size)

    return {
        "w": w,
        "delay": delay,
        "csr": csr,
        "by_type": {"EE": idxEE, "EI": idxEI, "IE": idxIE, "II": idxII},
    }
