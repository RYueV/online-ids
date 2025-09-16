import os, json
from typing import Dict
import numpy as np

import config
from reservoir.masks import build_indices
from reservoir.topology import build_topology
from reservoir.weights import assign_weights



# диапазоны задержек в DT
INTRA_DELAY = {"EE" : (1, 6), "EI" : (1, 4), "IE": (0, 2), "II" : (0, 1)}
ALERT_TO_THREAT = {"EE" : (1, 6), "EI" : (1, 4), "IE": (0, 2), "II" : (0, 2)}
CORE_TO_THREAT = {"EE" : (2, 10), "EI" : (2, 10), "IE": (0, 2), "II" : (0, 2)}
INTER_FALLBACK = {"EE" : (1, 6), "EI" : (1, 4), "IE": (0, 2), "II" : (0, 1)}

ST_MAP = {0:"EE", 1:"EI", 2:"IE", 3:"II"}


# формирование задержек в синапсах
def _setup_delays(
        rng:np.random.Generator,
        src:np.ndarray,
        dst:np.ndarray,
        stype:np.ndarray,
        masks:Dict[str,object]
    ):
    # количество синапсов
    M = int(src.size)
    delay = np.zeros(M, dtype=np.int16)
    # код сектора для каждого нерона, list[N]
    sector_of = masks["sector_of"]
    # коды секторов
    sec2code = masks["sec2code"]
    code2sec = {code:name for name,code in sec2code.items()}

    # коды секторов, к которым относятся pre-нейроны
    s_codes = sector_of[src]
    # коды секторов, к которым относятся post-нейроны
    t_codes = sector_of[dst]

    for st_code, st_name in ST_MAP.items():
        # булев массив, который хранит True для синапсов типа st_code (X->Y)
        m_type = (stype == st_code)
        if not np.any(m_type): continue

        idx = np.where(m_type)[0]
        # код сектора для каждого pre-X-нейрона 
        sc = s_codes[idx]
        # код сектора для каждого post-Y-нейрона 
        tc = t_codes[idx]

        # внутрисекторные связи
        same = (sc == tc)
        if np.any(same):
            low, high = INTRA_DELAY[st_name]
            delay[idx[same]] = rng.integers(low, high+1, size = int(np.sum(same)), dtype=np.int16)

        # межсекторные
        if np.any(~same):
            sub = idx[~same]
            s_names = np.array([code2sec[int(c)] for c in sc[~same]], dtype=object)
            t_names = np.array([code2sec[int(c)] for c in tc[~same]], dtype=object)

            m_alert_threat = (s_names=="alert") & np.isin(t_names, ("arp","syn","scan"))
            if np.any(m_alert_threat):
                low, high = ALERT_TO_THREAT[st_name]
                delay[sub[m_alert_threat]] = rng.integers(low, high+1, size=int(np.sum(m_alert_threat)), dtype=np.int16)

            m_core_threat = (s_names=="core") & np.isin(t_names, ("arp","syn","scan"))
            if np.any(m_core_threat):
                low, high = CORE_TO_THREAT[st_name]
                delay[sub[m_core_threat]] = rng.integers(low, high+1, size=int(np.sum(m_core_threat)), dtype=np.int16)

            m_rest = ~(m_alert_threat | m_core_threat)
            if np.any(m_rest):
                low, high = INTER_FALLBACK[st_name]
                delay[sub[m_rest]] = rng.integers(low, high+1, size=int(np.sum(m_rest)), dtype=np.int16)
    
    if np.unique(delay).size <= 1 and M > 0:
        delay ^= rng.integers(0, 2, size=M, dtype=np.int16)

    return delay



# формирование минимальных EE-кластеров в threat-секторах,
# если спектральный радиус матрицы весов в этих секторах 0
def _boost_threat_ee_clusters(
        rng:np.random.Generator,
        src:np.ndarray,
        dst:np.ndarray,
        stype:np.ndarray,
        masks:Dict[str,object],
        *,
        target_r_min:float=0.3,
        target_r_max:float=0.5,
        ee_mean_abs:float=0.15,
        max_cluster_size:int=10,
        max_new_edges_per_sector:int=48
    ):
    sec2code = masks["sec2code"]
    sector_of = masks["sector_of"]
    idxE_sector = masks.get("idxE_sector", None)

    if idxE_sector is None:
        idxE_sector = {}
        for name,code in sec2code.items():
            idxE_sector[name] = np.where((sector_of == code) & (masks["is_exc"]))[0]
        
    threat_names = [s for s in ("arp", "syn", "scan") if s in sec2code]
    if not threat_names:
        return src, dst, stype
    
    target_mid = 0.5 * (target_r_min + target_r_max)
    d_needed = int(np.clip(round(target_mid / max(1e-6, ee_mean_abs)), 1, 5))

    ee_mask = (stype == 0)
    ee_pairs = set(zip(src[ee_mask].tolist(), dst[ee_mask].tolist()))

    def _ext_ee_indices_for_sources(sec_code, sources):
        out = {int(u): [] for u in sources.tolist()}
        m = ee_mask & np.isin(src, sources)
        idxs = np.where(m)[0]
        for k in idxs:
            if sector_of[dst[k]] != sec_code:
                out[int(src[k])].append(int(k))
        return out
    
    new_src, new_dst, new_stype = [], [], []

    for s_name in threat_names:
        sec_code = sec2code[s_name]
        E_nodes = idxE_sector.get(s_name, np.array([], dtype=np.int64))

        if E_nodes.size < 2:
            continue

        S = min(max_cluster_size, E_nodes.size)
        cluster = rng.choice(E_nodes, size=S, replace=False)
        cluster_set = set(cluster.tolist())
        ext_by_src = _ext_ee_indices_for_sources(sec_code, cluster)

        for u in cluster:
            existing_targets = set()
            m_u = ee_mask & (src == u)
            for k in np.where(m_u)[0]:
                if sector_of[dst[k]] == sec_code:
                    existing_targets.add(int(dst[k]))
            need = max(0, d_needed - len(existing_targets))
            if need <= 0:
                continue
            candidates = [int(v) for v in cluster if v != u]
            rng.shuffle(candidates)
            rewired = 0
            if ext_by_src.get(int(u)):
                for v in candidates:
                    if rewired >= need:
                        break
                    if (u, v) in ee_pairs or v in existing_targets:
                        continue
                    k = ext_by_src[int(u)].pop() if ext_by_src[int(u)] else None
                    if k is None:
                        break
                    dst[k] = v
                    ee_pairs.add((int(u), int(v)))
                    existing_targets.add(int(v))
                    rewired += 1
            
            still = max(0, need - rewired)
            if still > 0 and max_new_edges_per_sector > 0:
                add_cnt = min(still, max_new_edges_per_sector)
                added = 0
                for v in candidates:
                    if added >= add_cnt:
                        break
                    if (u, v) in ee_pairs or v in existing_targets:
                        continue
                    new_src.append(int(u))
                    new_dst.append(int(v))
                    new_stype.append(0)  # EE
                    ee_pairs.add((int(u), int(v)))
                    existing_targets.add(int(v))
                    added += 1
                max_new_edges_per_sector -= added

    if new_src:
        src = np.concatenate([src, np.asarray(new_src, dtype=src.dtype)], axis=0)
        dst = np.concatenate([dst, np.asarray(new_dst, dtype=dst.dtype)], axis=0)
        stype = np.concatenate([stype, np.asarray(new_stype, dtype=stype.dtype)], axis=0)

    return src, dst, stype



def _ensure_threat_ee_circulant(
        rng:np.random.Generator,
        src:np.ndarray,
        dst:np.ndarray,
        stype:np.ndarray,
        masks:Dict[str, object],
        *,
        min_cluster_size:int=6,
        max_cluster_size:int=12,
        d_per_node:int=3,
        add_limit_per_sector:int=48
    ):
    sec2code = masks["sec2code"]
    sector_of = masks["sector_of"]
    is_exc = masks["is_exc"]

    idxE_sector = masks.get("idxE_sector")
    if idxE_sector is None:
        idxE_sector = {}
        for name, code in sec2code.items():
            idxE_sector[name] = np.where((sector_of == code) & (is_exc))[0]

    threat_names = [s for s in ("arp", "syn", "scan") if s in sec2code]
    if not threat_names:
        return src, dst, stype

    ee_mask = (stype == 0)
    ee_pairs = set(zip(src[ee_mask].tolist(), dst[ee_mask].tolist()))

    def _ext_ee_indices_for_sources(sec_code, sources, inner_set):
        out = {int(u): [] for u in sources.tolist()}
        m = ee_mask & np.isin(src, sources)
        idxs = np.where(m)[0]
        for k in idxs:
            if sector_of[dst[k]] != sec_code or (int(dst[k]) not in inner_set):
                out[int(src[k])].append(int(k))
        return out

    new_src, new_dst, new_stype = [], [], []

    for name in threat_names:
        code = sec2code[name]
        E_nodes = idxE_sector.get(name, np.array([], dtype=np.int64))
        if E_nodes.size < 2:
            continue

        S = E_nodes.size
        if S < min_cluster_size:
            S = E_nodes.size
        else:
            S = min(max_cluster_size, E_nodes.size)
        cluster = rng.choice(E_nodes, size=S, replace=False)
        cluster_set = set(int(u) for u in cluster)

        required_pairs: list[tuple[int,int]] = []
        for i, u in enumerate(cluster):
            for step in range(1, d_per_node + 1):
                v = int(cluster[(i + step) % S])
                if u == v:
                    continue
                required_pairs.append((int(u), v))

        ext_by_src = _ext_ee_indices_for_sources(code, cluster, cluster_set)
        added_in_sector = 0

        for (u, v) in required_pairs:
            if (u, v) in ee_pairs:
                continue

            k_list = ext_by_src.get(u, [])
            k = k_list.pop() if k_list else None
            if k is not None:
                dst[k] = v
                ee_pairs.add((u, v))
                continue

            if added_in_sector < add_limit_per_sector:
                new_src.append(u)
                new_dst.append(v)
                new_stype.append(0)
                ee_pairs.add((u, v))
                added_in_sector += 1

    if new_src:
        src = np.concatenate([src, np.asarray(new_src, dtype=src.dtype)], axis=0)
        dst = np.concatenate([dst, np.asarray(new_dst, dtype=dst.dtype)], axis=0)
        stype = np.concatenate([stype, np.asarray(new_stype, dtype=stype.dtype)], axis=0)

    return src, dst, stype


###
def _spectral_radius_power_local(src, dst, w, n, iters=30):
    if n <= 0 or src.size == 0:
        return 0.0
    rng = np.random.default_rng(123)
    x = rng.random(n, dtype=np.float64)
    x /= (np.linalg.norm(x) + 1e-12)
    for _ in range(max(1, iters)):
        y = np.zeros_like(x)
        np.add.at(y, dst, w * x[src])
        nrm = np.linalg.norm(y)
        if not np.isfinite(nrm) or nrm <= 1e-20:
            return 0.0
        x = y / nrm
    y = np.zeros_like(x)
    np.add.at(y, dst, w * x[src])
    num = float(np.dot(x, y))
    den = float(np.dot(x, x)) + 1e-12
    return float(abs(num / den))


def _raise_alert_ee_radius_by_weights(
    src:np.ndarray,
    dst:np.ndarray,
    stype:np.ndarray,
    w:np.ndarray,
    masks:Dict[str, object],
    target_alert_r:float=0.70,
    iters:int=30,
):
    sec2code = masks["sec2code"]
    is_exc = masks["is_exc"]
    sector_of = masks["sector_of"]

    if "alert" not in sec2code:
        return

    code = sec2code["alert"]
    E = np.where((sector_of == code) & (is_exc))[0]
    if E.size < 2:
        return

    idxEE = np.where(stype == 0)[0]
    if idxEE.size == 0:
        return

    m_alert_intra = np.isin(src[idxEE], E) & np.isin(dst[idxEE], E)
    ee_alert_idx = idxEE[m_alert_intra]
    if ee_alert_idx.size == 0:
        return

    map_local = -np.ones(is_exc.size, dtype=np.int64)
    map_local[E] = np.arange(E.size, dtype=np.int64)
    se_src = map_local[src[ee_alert_idx]]
    se_dst = map_local[dst[ee_alert_idx]]
    se_w = np.abs(w[ee_alert_idx].astype(np.float64, copy=False))
    r_now = _spectral_radius_power_local(se_src, se_dst, se_w, n=E.size, iters=iters)
    if r_now <= 0.0 or not np.isfinite(r_now):
        return

    ee_all_abs = np.abs(w[idxEE]).astype(np.float64, copy=False)
    sumA = float(np.sum(np.abs(w[ee_alert_idx])))
    sumB = float(np.sum(ee_all_abs) - sumA)
    C = sumA + sumB
    R = float(target_alert_r / r_now)

    denom = (C - R * sumA)
    if denom <= 1e-12:
        s = R
    else:
        s = (R * sumB) / denom

    s = float(np.clip(s, 1.05, 5.0))
    w[ee_alert_idx] *= s
    t = float(C / (sumB + s * sumA))
    w[idxEE] *= t
###


# построение и сохранение графа скрытого слоя
def build_hidden_graph(
        out_dir:str="reservoir/graph",
        basename:str="hidden_graph",
        seed:int=32,
        dedup_edges:bool=True
    ):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # разметка нейронов
    masks = build_indices(
        sector_order=None,
        detEI=True,
        rng=None,
        tau_m_by_sector=None
    )

    # топология скрытого слоя
    src, dst, stype = build_topology(
        masks=masks,
        rng=rng,
        dense_th=0.03,
        remove_self_loops=True,
        dedup=dedup_edges
    )

    ###
    src, dst, stype = _boost_threat_ee_clusters(
        rng, src, dst, stype, masks,
        target_r_min=0.4, 
        target_r_max=0.6,
        ee_mean_abs=0.15,
        max_cluster_size=10,
        max_new_edges_per_sector=48
    )

    src, dst, stype = _ensure_threat_ee_circulant(
        rng, src, dst, stype, masks,
        min_cluster_size=10,
        max_cluster_size=15,
        d_per_node=5,
        add_limit_per_sector=48
    )
    ###

    # веса
    w_pack = assign_weights(
        src, dst, stype, masks,
        rng=rng,
        e_mean=0.15, e_std=0.02,
        i_mean=0.36, i_std=0.04,
        ee_radius_target=0.95,
        i_to_e_ratio=None,
        post_abs_max=None,
        delay_range=None,
        build_csr=False
    )
    w = w_pack["w"].astype(np.float32, copy=False)

    ###
    _raise_alert_ee_radius_by_weights(
        src=src, dst=dst, stype=stype, w=w, masks=masks,
        target_alert_r=0.70, iters=30
    )
    ###

    # задержки
    delay = _setup_delays(rng, src, dst, stype, masks)

    # сохранение
    npz_path = os.path.join(out_dir, f"{basename}.npz")
    np.savez_compressed(
        npz_path,
        src=src.astype(np.int32, copy=False),
        dst=dst.astype(np.int32, copy=False),
        stype=stype.astype(np.int32, copy=False),
        w=w,
        delay=delay.astype(np.int16, copy=False),
        is_exc=masks["is_exc"].astype(np.bool_, copy=False),
        sector_of=masks["sector_of"].astype(np.int16, copy=False),
    )

    meta = {
        "sec2code": masks["sec2code"],
        "seed": seed,
        "dedup_edges": dedup_edges,
        "counts": {
            "n_neurons": int(masks["is_exc"].size),
            "n_edges": int(src.size),
            "by_type": {
                "EE": int(np.sum(stype==0)),
                "EI": int(np.sum(stype==1)),
                "IE": int(np.sum(stype==2)),
                "II": int(np.sum(stype==3)),
            }
        },
        "delay_policy_ms": {
            "intra": INTRA_DELAY,
            "alert_to_threat": ALERT_TO_THREAT,
            "core_to_threat": CORE_TO_THREAT,
            "inter_fallback": INTER_FALLBACK,
            "dt_ms": int(round(config.DT * 1000.0)),
        },
        "weights_target_mean_abs": {"EE":0.15,"EI":0.15,"IE":0.33,"II":0.33}
    }
    json_path = os.path.join(out_dir, f"{basename}_meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "src": src, 
        "dst": dst, 
        "stype": stype, 
        "w": w, 
        "delay": delay,
        "masks": masks, 
        "paths": {"npz": npz_path, "json": json_path}
    }



if __name__ == "__main__":
    build_hidden_graph()