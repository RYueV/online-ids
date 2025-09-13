# Отчёт E/I-калькулятора

## Размеры пулов

- **core**: N=240, nE=192, nI=48
- **alert**: N=78, nE=62, nI=16
- **arp**: N=42, nE=34, nI=8
- **scan**: N=47, nE=38, nI=9
- **syn**: N=23, nE=18, nI=5

## Исходные целевые параметры

- rE=2.0 Гц, rI=6.0 Гц
- |w|: EE=0.6, EI=0.6, IE=1.0, II=1.0

## Степени (k) и вероятности (p) по блокам

### Сектор core
- EE: k_tot=13.44
  - core-> core: k=13.44, p=0.0700
  - alert-> core: k=0.00, p=0.0000
  - arp-> core: k=0.00, p=0.0000
  - scan-> core: k=0.00, p=0.0000
  - syn-> core: k=0.00, p=0.0000
- IE: k_tot=5.65
  - core-> core: k=5.65, p=0.1176
  - alert-> core: k=0.00, p=0.0000
  - arp-> core: k=0.00, p=0.0000
  - scan-> core: k=0.00, p=0.0000
  - syn-> core: k=0.00, p=0.0000
- EI: k_tot=17.54
  - core-> core: k=17.54, p=0.0914
  - alert-> core: k=0.00, p=0.0000
  - arp-> core: k=0.00, p=0.0000
  - scan-> core: k=0.00, p=0.0000
  - syn-> core: k=0.00, p=0.0000
- II: k_tot=3.16
  - core-> core: k=3.16, p=0.0658
  - alert-> core: k=0.00, p=0.0000
  - arp-> core: k=0.00, p=0.0000
  - scan-> core: k=0.00, p=0.0000
  - syn-> core: k=0.00, p=0.0000

### Сектор alert
- EE: k_tot=7.14
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=7.14, p=0.1152
  - arp-> alert: k=0.00, p=0.0000
  - scan-> alert: k=0.00, p=0.0000
  - syn-> alert: k=0.00, p=0.0000
- IE: k_tot=2.57
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=2.57, p=0.1607
  - arp-> alert: k=0.00, p=0.0000
  - scan-> alert: k=0.00, p=0.0000
  - syn-> alert: k=0.00, p=0.0000
- EI: k_tot=8.33
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=8.33, p=0.1344
  - arp-> alert: k=0.00, p=0.0000
  - scan-> alert: k=0.00, p=0.0000
  - syn-> alert: k=0.00, p=0.0000
- II: k_tot=1.33
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=1.33, p=0.0833
  - arp-> alert: k=0.00, p=0.0000
  - scan-> alert: k=0.00, p=0.0000
  - syn-> alert: k=0.00, p=0.0000

### Сектор arp
- EE: k_tot=3.23
  - core-> arp: k=0.81, p=0.0042
  - alert-> arp: k=1.78, p=0.0287
  - arp-> arp: k=0.65, p=0.0190
  - scan-> arp: k=0.00, p=0.0000
  - syn-> arp: k=0.00, p=0.0000
- IE: k_tot=1.23
  - core-> arp: k=0.37, p=0.0077
  - alert-> arp: k=0.61, p=0.0384
  - arp-> arp: k=0.25, p=0.0307
  - scan-> arp: k=0.00, p=0.0000
  - syn-> arp: k=0.00, p=0.0000
- EI: k_tot=4.29
  - core-> arp: k=1.71, p=0.0089
  - alert-> arp: k=1.71, p=0.0276
  - arp-> arp: k=0.86, p=0.0252
  - scan-> arp: k=0.00, p=0.0000
  - syn-> arp: k=0.00, p=0.0000
- II: k_tot=0.64
  - core-> arp: k=0.06, p=0.0013
  - alert-> arp: k=0.06, p=0.0040
  - arp-> arp: k=0.51, p=0.0643
  - scan-> arp: k=0.00, p=0.0000
  - syn-> arp: k=0.00, p=0.0000

### Сектор scan
- EE: k_tot=5.17
  - core-> scan: k=1.29, p=0.0067
  - alert-> scan: k=2.84, p=0.0459
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=1.03, p=0.0272
  - syn-> scan: k=0.00, p=0.0000
- IE: k_tot=1.97
  - core-> scan: k=0.59, p=0.0123
  - alert-> scan: k=0.98, p=0.0614
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=0.39, p=0.0437
  - syn-> scan: k=0.00, p=0.0000
- EI: k_tot=6.67
  - core-> scan: k=2.67, p=0.0139
  - alert-> scan: k=2.67, p=0.0430
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=1.33, p=0.0351
  - syn-> scan: k=0.00, p=0.0000
- II: k_tot=1.00
  - core-> scan: k=0.10, p=0.0021
  - alert-> scan: k=0.10, p=0.0062
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=0.80, p=0.0889
  - syn-> scan: k=0.00, p=0.0000

### Сектор syn
- EE: k_tot=3.88
  - core-> syn: k=0.97, p=0.0051
  - alert-> syn: k=2.13, p=0.0344
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=0.78, p=0.0431
- IE: k_tot=1.47
  - core-> syn: k=0.44, p=0.0092
  - alert-> syn: k=0.74, p=0.0461
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=0.29, p=0.0590
- EI: k_tot=5.00
  - core-> syn: k=2.00, p=0.0104
  - alert-> syn: k=2.00, p=0.0323
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=1.00, p=0.0556
- II: k_tot=0.75
  - core-> syn: k=0.07, p=0.0016
  - alert-> syn: k=0.07, p=0.0047
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=0.60, p=0.1200
