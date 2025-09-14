"""
Топология скрытого слоя
"""
import numpy as np
from typing import Dict, Optional, Tuple
from config import P_INTER_BY_PAIR, P_INTRA_BY_SECTOR

# коды типов связей
STYPE = {
    "EE" : np.int8(0), "EI" : np.int8(1),
    "IE" : np.int8(2), "II" : np.int8(3)
}


# вспомогательный метод для генерации "пустых" ребер
def _empty_edges():
    return (
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32)
    )


# генерация ребер (E/I связей) между двумя наборами нейронов (pre->post)
def _sample_block(
      *,
      # индексы пресинаптических нейронов
      pre_idx:np.ndarray,
      # индексы постсинаптических нейронов
      post_idx:np.ndarray,
      # доля связей типа stype_code
      p:float,
      # код типа связи
      stype_code:np.int8,
      # генератор ГПСЧ
      rng:np.random.Generator,
      # порог для построения связей без полного перебора
      dense_th:float,
      # нужно ли убирать самопетли
      remove_self_loops:bool  
    )->Tuple[np.ndarray,np.ndarray,np.ndarray]:

    # количество pre-нейронов
    n_pre = pre_idx.size
    # количество post-нейронов
    n_post = post_idx.size

    # проверка корректности исходных данных
    if p <= 0.0 or n_pre == 0 or n_post == 0:
        return _empty_edges()
    
    # построение с помощью полного перебора при малой доле связей
    if p < dense_th:
        src_list = []
        dst_list = []

        # выбор k ~ Binom(n_post,p) уникальных получателей для каждого источника
        for pre in pre_idx:
            k = rng.binomial(n=n_post, p=p)
            if k == 0: continue

            # удаление петлевых связей
            if remove_self_loops and n_pre == n_post and pre_idx is post_idx:
                if n_post <= 1: continue
                post = rng.choice(post_idx, size=k, replace=False)
                if pre in post:
                    post = post[post != pre]
                    need = k - post.size
                    if need > 0 and (n_post - 1) >= need:
                        pool = post_idx[post_idx != pre]
                        add = rng.choice(pool, size=need, replace=False)
                        post = np.concatenate([post, add])
            else:
                post = rng.choice(post_idx, size=k, replace=False)

            # индекс получателя (k раз является источником)
            src_list.extend([int(pre)]*k)
            # индексы целей
            dst_list.extend(map(int, post)) 

        if not src_list:
            return _empty_edges()
        
        # list -> np.ndarray
        src = np.fromiter(src_list, dtype=np.int32, count=len(src_list))
        dst = np.fromiter(dst_list, dtype=np.int32, count=len(dst_list))
        st = np.full(src.size, stype_code, dtype=np.int8)

        if remove_self_loops and pre_idx is post_idx:
            mask = src != dst
            if not np.all(mask):
                src = src[mask]
                dst = dst[mask]
                st = st[mask]
        
        return src, dst, st
    

    # построение блоками для экономии памяти
    batch = 1024 if n_pre >= 2048 else max(64, n_pre)
    src_list = []
    dst_list = []

    for start in range(0, n_pre, batch):
        stop = min(start + batch, n_pre)
        cur_pre = pre_idx[start:stop]
        mask = rng.random((cur_pre.size, n_post)) < p

        if remove_self_loops and pre_idx is post_idx:
            for i, pre in enumerate(cur_pre):
                pos = np.searchsorted(post_idx, pre)
                if pos < n_post and post_idx[pos] == pre:
                    mask[i, pos] = False 
    
        rows, cols = np.nonzero(mask)
        if rows.size == 0: continue
        src_list.append(cur_pre[rows])
        dst_list.append(post_idx[cols])

    if not src_list:
        return _empty_edges()
    
    src = np.concatenate(src_list).astype(np.int32, copy=False)
    dst = np.concatenate(dst_list).astype(np.int32, copy=False)
    st = np.full(src.size, stype_code, dtype=np.int8)
    return src, dst, st
    




# построение графа внутрисекторных и межсекторных связей
def build_topology(
        # словарь с разметкой нейронов
        masks:Dict[str, object],
        *,
        # объект ГПСЧ
        rng:Optional[np.random.Generator]=None,
        # порог для построения связей без полного перебора
        dense_th:float=0.03,
        # нужно ли убирать самопетли (для внутрисекторных связей)
        remove_self_loops:bool=True,
        # нужно ли убирать дубликаты ребер
        dedup:bool=False        
    )->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Возвращает список троек (pre, post, stype)
        pre - индексы пресинаптических нейронов
        post - индексы постсинаптических нейронов
        stype - код синапса: 0=EE, 1=EI, 2=IE, 3=II
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # глобальные индексы pre нейронов
    src_parts = []
    # глобальные индексы post нейронов
    dst_parts = []
    # типы связей
    st_parts = []

    # построение внутрисекторных связей
    # (обход по E-нейронам каждого сектора s_name)
    for s_name, idxE in masks["idxE_sector"].items():
        # индексы I-нейронов сектора s_name
        idxI = masks["idxI_sector"][s_name]

        p_local = P_INTRA_BY_SECTOR.get(s_name, {})
        pEE = float(p_local.get("EE", 0.0))
        pEI = float(p_local.get("EI", 0.0))
        pIE = float(p_local.get("IE", 0.0))
        pII = float(p_local.get("II", 0.0))

        # E->E
        src, dst, st = _sample_block(
            pre_idx=idxE,
            post_idx=idxE,
            p=pEE,
            stype_code=STYPE["EE"],
            rng=rng,
            dense_th=dense_th,
            remove_self_loops=remove_self_loops
        )
        if src.size:
            src_parts.append(src); dst_parts.append(dst); st_parts.append(st)

        # E->I
        src, dst, st = _sample_block(
            pre_idx=idxE,
            post_idx=idxI,
            p=pEI,
            stype_code=STYPE["EI"],
            rng=rng,
            dense_th=dense_th,
            remove_self_loops=remove_self_loops
        )
        if src.size:
            src_parts.append(src); dst_parts.append(dst); st_parts.append(st)

        # I->E
        src, dst, st = _sample_block(
            pre_idx=idxI,
            post_idx=idxE,
            p=pIE,
            stype_code=STYPE["IE"],
            rng=rng,
            dense_th=dense_th,
            remove_self_loops=remove_self_loops
        )
        if src.size:
            src_parts.append(src); dst_parts.append(dst); st_parts.append(st)

        # I->I
        src, dst, st = _sample_block(
            pre_idx=idxI,
            post_idx=idxI,
            p=pII,
            stype_code=STYPE["II"],
            rng=rng,
            dense_th=dense_th,
            remove_self_loops=remove_self_loops
        )
        if src.size:
            src_parts.append(src); dst_parts.append(dst); st_parts.append(st)

    # межсекторные связи
    for (src_sec, dst_sec), probs in P_INTER_BY_PAIR.items():
        Esrc = masks["idxE_sector"][src_sec]
        Isrc = masks["idxI_sector"][src_sec]
        Edst = masks["idxE_sector"][dst_sec]
        Idst = masks["idxI_sector"][dst_sec]

        pEE = float(probs.get("EE", 0.0))
        pEI = float(probs.get("EI", 0.0))
        pIE = float(probs.get("IE", 0.0))
        pII = float(probs.get("II", 0.0))

        # E->E
        if pEE > 0.0 and Esrc.size and Edst.size:
            src, dst, st = _sample_block(
                pre_idx=Esrc,
                post_idx=Edst,
                p=pEE,
                stype_code=STYPE["EE"],
                rng=rng,
                dense_th=dense_th,
                remove_self_loops=False
            )
            if src.size:
                src_parts.append(src); dst_parts.append(dst); st_parts.append(st)
        
        # E->I
        if pEI > 0.0 and Esrc.size and Idst.size:
            src, dst, st = _sample_block(
                pre_idx=Esrc,
                post_idx=Idst,
                p=pEI,
                stype_code=STYPE["EI"],
                rng=rng,
                dense_th=dense_th,
                remove_self_loops=False
            )
            if src.size:
                src_parts.append(src); dst_parts.append(dst); st_parts.append(st)

        # I->E
        if pIE > 0.0 and Isrc.size and Edst.size:
            src, dst, st = _sample_block(
                pre_idx=Isrc, post_idx=Edst, p=pIE,
                stype_code=STYPE["IE"], rng=rng,
                dense_th=dense_th, remove_self_loops=False
            )
            if src.size:
                src_parts.append(src); dst_parts.append(dst); st_parts.append(st)

        # I->I
        if pII > 0.0 and Isrc.size and Idst.size:
            src, dst, st = _sample_block(
                pre_idx=Isrc, post_idx=Idst, p=pII,
                stype_code=STYPE["II"], rng=rng,
                dense_th=dense_th, remove_self_loops=False
            )
            if src.size:
                src_parts.append(src); dst_parts.append(dst); st_parts.append(st)
        
    # склейка
    if src_parts:
        src = np.concatenate(src_parts).astype(np.int32, copy=False)
        dst = np.concatenate(dst_parts).astype(np.int32, copy=False)
        stype = np.concatenate(st_parts).astype(np.int32, copy=False)
    else:
        src = np.empty(0, dtype=np.int32)
        dst = np.empty(0, dtype=np.int32)
        stype = np.empty(0, dtype=np.int32)

    # удаление дубликатов
    if dedup and src.size:
        key = (stype.astype(np.int64) << 56) | (src.astype(np.int64) << 28) | dst.astype(np.int64)
        uniq, idx = np.unique(key, return_index=True)
        src = src[idx]; dst = dst[idx]; stype = stype[idx]
    
    return src, dst, stype
    


