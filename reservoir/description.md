- is_exc : bool[N]  
  True если нейрон возбуждающий, False если тормозный

- sector_of : int16[N]  
  код сектора для каждого нейрона

- sec2code : dict {str -> int}  
  словарь с кодами секторов

- idxE_sector : dict {sector_name -> np.ndarray}  
  индексы возбуждающих нейронов по секторам

- idxI_sector : dict {sector_name -> np.ndarray}  
  индексы тормозных нейронов по секторам

- sens_of : float32[N]  
  чувствительность нейрона к модуляции LC

- tau_m0 : float32[N]  
  базовое значение tau мембраны нейрона

- src : int32[M]  
  глобальный индекс нейрона источника для каждого ребра

- dst : int32[M]  
  глобальный индекс нейрона приемника для каждого ребра

- stype : int32[M]  
  тип связи ребра, код 0 = EE, 1 = EI, 2 = IE, 3 = II

- w : float32[M]  
  вес синапса для каждого ребра, положительный для возбуждающих, отрицательный для тормозных

- delay : int16[M]  
  задержка синапса в шагах дискретизации

- csr : dict  
  структура для быстрого вычисления рекуррентного тока, содержит indptr, indices, data

- V : float32[N]  
  мембранный потенциал каждого нейрона

- ref_left : float32[N]  
  оставшееся время рефрактерного периода

- vth0 : float32[N]  
  базовое значение порога возбуждения

- vth : float32[N]  
  текущее значение порога возбуждения

- tau_m : float32[N]  
  текущее значение tau мембраны

- spikes : int8[N]  
  бинарный индикатор спайка на шаге, 1 если был спайк

- in_curr : float32[N]  
  входной ток от внешних признаков

- xE : float32[N]  
  следы возбуждающих синапсов

- xI : float32[N]  
  следы тормозных синапсов

- w_adapt : float32[N]  
  адаптационный ток нейрона

- alpha : float32  
  уровень тревоги LC

- lc_ref : float32  
  состояние рефрактерности LC

- g_inh_base : float32[N]  
  базовый уровень дезингибиции входа

- g_inh_eff : float32[N]  
  текущий эффективный уровень дезингибиции

- feat2neurons : dict {feature_id -> np.ndarray}  
  проекция признаков на индексы E нейронов

- feat_gain : dict {feature_id -> float}  
  коэффициент усиления для признака

- alert_feat_ids : list[int]  
  список feature id относящихся к тревожным признакам

- safety_feat_ids : list[int]  
  список feature id относящихся к безопасным признакам

- threat_feat_ids : list[int]  
  список feature id относящихся к атакующим признакам

- masks : Dict[str,object]
  словарь с разметкой нейронов
  {
    "idx_sector" : idx_sector,
    "idxE_sector" : idxE_sector,
    "idxI_sector" : idxI_sector,
    "is_exc" : is_exc,
    "sector_of" : sector_of,
    "sens_of" : sens_of,
    "tau_m0" : tau_m0,
    "sec2code" : sec2code
  }