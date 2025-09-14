# Отчёт E/I-калькулятора

## Размеры пулов

- **core**: N=240, nE=192, nI=48
- **alert**: N=78, nE=62, nI=16
- **arp**: N=42, nE=34, nI=8
- **scan**: N=47, nE=38, nI=9
- **syn**: N=23, nE=18, nI=5

## Исходные целевые параметры

- rE=8.0 Гц, rI=18.0 Гц
- |w|: EE=0.15, EI=0.15, IE=0.33, II=0.33

## Степени (k) и вероятности (p) по блокам

### Сектор core
- EE: k_tot=13.44
  - core-> core: k=13.44, p=0.0700
  - alert-> core: k=0.00, p=0.0000
  - arp-> core: k=0.00, p=0.0000
  - scan-> core: k=0.00, p=0.0000
  - syn-> core: k=0.00, p=0.0000
- IE: k_tot=5.70
  - core-> core: k=5.70, p=0.1188
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
- II: k_tot=3.19
  - core-> core: k=3.19, p=0.0665
  - alert-> core: k=0.00, p=0.0000
  - arp-> core: k=0.00, p=0.0000
  - scan-> core: k=0.00, p=0.0000
  - syn-> core: k=0.00, p=0.0000

### Сектор alert
- EE: k_tot=6.25
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=6.25, p=0.1008
  - arp-> alert: k=0.00, p=0.0000
  - scan-> alert: k=0.00, p=0.0000
  - syn-> alert: k=0.00, p=0.0000
- IE: k_tot=2.27
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=2.27, p=0.1420
  - arp-> alert: k=0.00, p=0.0000
  - scan-> alert: k=0.00, p=0.0000
  - syn-> alert: k=0.00, p=0.0000
- EI: k_tot=8.33
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=8.33, p=0.1344
  - arp-> alert: k=0.00, p=0.0000
  - scan-> alert: k=0.00, p=0.0000
  - syn-> alert: k=0.00, p=0.0000
- II: k_tot=1.35
  - core-> alert: k=0.00, p=0.0000
  - alert-> alert: k=1.35, p=0.0842
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
- IE: k_tot=1.24
  - core-> arp: k=0.37, p=0.0078
  - alert-> arp: k=0.62, p=0.0388
  - arp-> arp: k=0.25, p=0.0310
  - scan-> arp: k=0.00, p=0.0000
  - syn-> arp: k=0.00, p=0.0000
- EI: k_tot=4.29
  - core-> arp: k=1.71, p=0.0089
  - alert-> arp: k=1.71, p=0.0276
  - arp-> arp: k=0.86, p=0.0252
  - scan-> arp: k=0.00, p=0.0000
  - syn-> arp: k=0.00, p=0.0000
- II: k_tot=0.65
  - core-> arp: k=0.06, p=0.0014
  - alert-> arp: k=0.06, p=0.0041
  - arp-> arp: k=0.52, p=0.0649
  - scan-> arp: k=0.00, p=0.0000
  - syn-> arp: k=0.00, p=0.0000

### Сектор scan
- EE: k_tot=5.17
  - core-> scan: k=1.29, p=0.0067
  - alert-> scan: k=2.84, p=0.0459
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=1.03, p=0.0272
  - syn-> scan: k=0.00, p=0.0000
- IE: k_tot=1.99
  - core-> scan: k=0.60, p=0.0124
  - alert-> scan: k=0.99, p=0.0620
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=0.40, p=0.0441
  - syn-> scan: k=0.00, p=0.0000
- EI: k_tot=6.67
  - core-> scan: k=2.67, p=0.0139
  - alert-> scan: k=2.67, p=0.0430
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=1.33, p=0.0351
  - syn-> scan: k=0.00, p=0.0000
- II: k_tot=1.01
  - core-> scan: k=0.10, p=0.0021
  - alert-> scan: k=0.10, p=0.0063
  - arp-> scan: k=0.00, p=0.0000
  - scan-> scan: k=0.81, p=0.0898
  - syn-> scan: k=0.00, p=0.0000

### Сектор syn
- EE: k_tot=3.88
  - core-> syn: k=0.97, p=0.0051
  - alert-> syn: k=2.13, p=0.0344
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=0.78, p=0.0431
- IE: k_tot=1.49
  - core-> syn: k=0.45, p=0.0093
  - alert-> syn: k=0.74, p=0.0465
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=0.30, p=0.0596
- EI: k_tot=5.00
  - core-> syn: k=2.00, p=0.0104
  - alert-> syn: k=2.00, p=0.0323
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=1.00, p=0.0556
- II: k_tot=0.76
  - core-> syn: k=0.08, p=0.0016
  - alert-> syn: k=0.08, p=0.0047
  - arp-> syn: k=0.00, p=0.0000
  - scan-> syn: k=0.00, p=0.0000
  - syn-> syn: k=0.61, p=0.1212
