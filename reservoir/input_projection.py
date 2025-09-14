from typing import Optional, Dict, Sequence, Tuple
import numpy as np
from config import (
    FEATURE_GROUPS,
    FEATURE_TARGET_SECTOR,
    INPUT_FANOUT,
    FEATURE_GAIN,
    INPUT_DETERMINISTIC,
    INPUT_RNG_SEED,
    SAFETY_FEATURE_IDS
)


# отображение feature_id -> E-нейроны соответствующего сектора
def build_input_projection(
        # разметка нейронов скрытого слоя по секторам
        masks:Dict[str, object],
        *,
        # детерменированное отображение
        det:Optional[bool]=None,
        # объект ГПСЧ
        rng:Optional[np.random.Generator]=None
    )->Dict[str, object]:
    if det is None:
        det = INPUT_DETERMINISTIC
    if not det and rng is None:
        rng = np.random.default_rng(INPUT_RNG_SEED)
    
    # словарь id признака -> id скрытого нейрона
    feat2neurons = {}
    # словарь id признака -> усиление входа
    feat_gain = {}

    # group - наименование группы входных признаков
    # ids - индексы признаков, относящихся к этой группе
    for group, ids in FEATURE_GROUPS.items():
        if not ids: continue
        # наименование сектора, куда должны поступать входы типа group
        sector = FEATURE_TARGET_SECTOR[group]
        # E-нейроны этого сектора
        idxE = masks["idxE_sector"][sector]
        # количество E-нейронов
        nE = int(idxE.size)
        if nE == 0: continue

        # сколько E-нейронов в sector активирует признак из group
        fanout = int(INPUT_FANOUT.get(group, 1))
        fanout = max(1, min(fanout, nE))
        # усиление входа для группы признаков group
        gain = float(FEATURE_GAIN.get(group, 1.0))
        
        # детерминированный (кольцевой) выбор связей feature -> E-neuron
        if det:
            for feature_id in ids:
                start = int(feature_id) % nE
                if start + fanout <= nE:
                    sel = idxE[start:start+fanout]
                else:
                    k1 = nE - start
                    sel = np.concatenate([idxE[start:], idxE[:fanout-k1]])
                feat2neurons[int(feature_id)] = sel.astype(np.int32, copy=True)
                feat_gain[int(feature_id)] = gain
        else:
            for feature_id in ids:
                sel = rng.choice(idxE, size=fanout, replace=False).astype(np.int32)
                feat2neurons[int(feature_id)] = sel
                feat_gain[int(feature_id)] = gain
    
    # индексы "тревожных" признаков для LC
    alert_ids = set(int(x) for x in FEATURE_GROUPS.get("alert", set()))
    # индексы признаков-сигналов о норме
    safety_ids = set(int(x) for x in SAFETY_FEATURE_IDS)
    # индексы признаков-атак
    threat_feat_ids = {
        "arp" : set(int(x) for x in FEATURE_GROUPS.get("arp", set())),
        "syn" : set(int(x) for x in FEATURE_GROUPS.get("syn", set())),
        "scan" : set(int(x) for x in FEATURE_GROUPS.get("scan", set())),
    }

    return {
        "feat2neurons" : feat2neurons,
        "feat_gain" : feat_gain,
        "alert_feat_ids" : alert_ids,
        "safety_feat_ids" : safety_ids,
        "threat_feat_ids" : threat_feat_ids
    }
    


# обработка входов за шаг
def apply_inputs(
        # индексы входов, сработавших за шаг
        step_feature_ids:Sequence[int],
        # состояние входного слоя
        inp_state:Dict[str,object],
        # состояние скрытого слоя 
        hid_state:Dict[str,object]
    )->Tuple[int,int]:
    # текущее значение входного тока
    in_curr = hid_state["in_curr"]
    # проекция признак -> скрытый E-нейрон
    feat2neurons = inp_state["feat2neurons"]
    # усиление входов
    feat_gain = inp_state["feat_gain"]
    # входы, привязанные к alert сектору
    alert_ids = inp_state["alert_feat_ids"]
    # входы, сигнализирующие о безопасном состоянии сети
    safety_ids = inp_state["safety_feat_ids"]

    # количество "опасных" сигналов за шаг
    n_alert = 0
    # количество "безопасных" сигналов за шаг
    n_safety = 0

    for feature_id in step_feature_ids:
        fid = int(feature_id)
        # индексы E-нейронов, куда поступает сигнал от fid
        sel = feat2neurons.get(fid)
        if sel is not None and sel.size:
            # увеличение входного тока с учетом усиления для fid
            in_curr[sel] += feat_gain.get(fid, 1.0)
        if fid in alert_ids:
            n_alert += 1
        if fid in safety_ids:
            n_safety += 1
        
    return n_alert, n_safety


# счетчик (для отладки)
def count_alert_and_safety(
        step_feature_ids:Sequence[int],
        inp_state:Dict[str,object]
    )->Tuple[int,int]:
    alert_ids = inp_state["alert_feat_ids"]
    safety_ids = inp_state["safety_feat_ids"]

    n_alert = 0
    n_safety = 0
    for feature_id in step_feature_ids:
        fid = int(feature_id)
        if fid in alert_ids:
            n_alert += 1
        if fid in safety_ids:
            n_safety += 1
    
    return n_alert, n_safety