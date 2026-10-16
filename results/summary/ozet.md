Friedman χ² = 103.46, p = 3.68e-17 (k = 12, N = 15)

Ortalama sıra (R²):
Model
SCWGI           2.47
EM              2.73
SCWGI-T         3.07
SAINT           4.27
TabDDPM         6.07
HyperImpute     6.20
MissForest      7.40
GAIN            7.87
SoftImpute      8.13
Ortalama        9.20
MICE            9.73
MIWAE          10.87

R² birincilik (koşul):
SCWGI          1
SCWGI-T        3
EM             6
SAINT          3
HyperImpute    0
TabDDPM        0
MissForest     0
GAIN           2
SoftImpute     0
MICE           0
MIWAE          0
Ortalama       0

MAE birincilik (koşul):
SCWGI          0
SCWGI-T        3
EM             0
SAINT          3
HyperImpute    4
TabDDPM        0
MissForest     3
GAIN           2
SoftImpute     0
MICE           0
MIWAE          0
Ortalama       0

SCWGI'ye karşı Wilcoxon + Holm (15 koşul):
      Rakip  SCWGI daha iyi (koşul)  Ort. ΔR²        p   p_Holm
    TabDDPM                      15    0.1456 0.000061 0.000610
HyperImpute                      15    0.2008 0.000061 0.000610
 SoftImpute                      15    0.2322 0.000061 0.000610
       MICE                      15    0.5988 0.000061 0.000610
   Ortalama                      15    0.5584 0.000061 0.000610
      MIWAE                      15    0.7195 0.000061 0.000610
 MissForest                      13    0.3035 0.000305 0.001221
      SAINT                      12    0.0734 0.004272 0.012817
       GAIN                      13    0.1718 0.055359 0.110718
         EM                       7   -0.0005 0.761536 0.761536

SCWGI'ye karşı Wilcoxon + Holm (MNAR hariç 12 koşul):
      Rakip  SCWGI daha iyi (koşul)  Ort. ΔR²        p   p_Holm
    TabDDPM                      12    0.1364 0.000488 0.004883
HyperImpute                      12    0.1124 0.000488 0.004883
 SoftImpute                      12    0.2106 0.000488 0.004883
       GAIN                      12    0.3028 0.000488 0.004883
       MICE                      12    0.4771 0.000488 0.004883
      MIWAE                      12    0.7033 0.000488 0.004883
   Ortalama                      12    0.5780 0.000488 0.004883
 MissForest                      10    0.1366 0.002441 0.007324
      SAINT                       9    0.0593 0.034180 0.068359
         EM                       6    0.0130 0.339355 0.339355

Endüktif grup birincilik:
SCWGI    12
SAINT     3

Endüktif grup ortalama sıra:
Model
SCWGI      1.20
SAINT      2.07
TabDDPM    2.73

Transdüktif grup ortalama sıra:
Model
EM             2.00
SCWGI-T        2.07
HyperImpute    4.20
MissForest     4.73
GAIN           5.40
SoftImpute     5.40
Ortalama       6.40
MICE           6.93
MIWAE          7.87

Transdüktif grup birincilik:
EM            6
SCWGI-T       4
GAIN          3
MissForest    2

SCWGI-T'ye karşı Wilcoxon + Holm (transdüktif grup):
      Rakip  SCWGI-T daha iyi (koşul)  Ort. ΔR²        p   p_Holm
   Ortalama                        15    0.5163 0.000061 0.000488
      MIWAE                        15    0.6775 0.000061 0.000488
       MICE                        15    0.5567 0.000061 0.000488
 SoftImpute                        15    0.1901 0.000061 0.000488
 MissForest                        13    0.2614 0.000305 0.001221
HyperImpute                        14    0.1588 0.000427 0.001282
       GAIN                        12    0.1298 0.207764 0.415527
         EM                         5   -0.0426 0.389404 0.415527

SCWGI-T − SCWGI (eşleştirilmiş, tohum düzeyi):
                      ΔR²      SE   n
Grup                                 
BreastCancer MCAR  0.0036  0.0009  15
CWRU MCAR          0.0032  0.0008   5
Diabetes MAR      -0.0266  0.0186  10
Fertility MCAR    -0.0108  0.0047  15
Fertility MNAR    -0.1913  0.0240  15
GoldPrice MCAR     0.0049  0.0003  15

Tohum düzeyinde R² birincilik (75):
SCWGI-T       21
EM            19
SAINT         11
GAIN          10
SCWGI          8
MissForest     3
MICE           2
TabDDPM        1

MNAR MAE (düşük = iyi):
Dataset     Fertility              
Mechanism        MNAR              
Rate              %10    %20    %30
Model                              
SCWGI           1.972  1.712  1.605
SCWGI-T         2.040  1.786  1.689
EM              1.989  1.697  1.663
SAINT           1.997  1.742  1.630
HyperImpute     2.012  1.581  1.594
TabDDPM         1.976  1.671  1.579
MissForest      1.975  1.850  1.738
GAIN            1.786  1.591  1.574
SoftImpute      2.048  1.835  1.695
MICE            2.024  1.875  1.898
MIWAE           2.094  1.901  1.732
Ortalama        2.068  1.909  1.778

Medyan süre (s):
Dataset      BreastCancer   CWRU  Diabetes  Fertility  GoldPrice
Model                                                           
EM                    2.3    4.5       0.1        0.1      473.9
GAIN                  7.8    7.4       5.4        5.3       13.8
HyperImpute          41.5   65.8      40.8       43.8      564.1
MICE                 46.7   13.7       6.7        9.0      299.0
MIWAE                38.7  115.7      16.8        3.4      256.2
MissForest           57.9  167.0       6.3        8.5      489.9
Ortalama              0.0    0.0       0.0        0.0        0.0
SAINT                43.0  283.9      24.4       10.1      586.3
SCWGI                16.9   61.7      10.2        1.7      394.1
SCWGI-T              36.8   99.8      22.2        5.1      479.9
SoftImpute           25.1   59.8      13.5       10.5      108.1
TabDDPM               9.3   41.5      18.4       11.3       37.2
