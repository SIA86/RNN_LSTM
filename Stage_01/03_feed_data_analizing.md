# Исследования влияния характера котировок передаваемых для тренировки НС на точность предсказаний и их универсальность
В ходе исследования необходимо установить насколько сильно зависит точность предсказаний от динамики изменения параметров данных (тренд, волатильность и т.п.) передаваемых для обучения НС. Необходимо выяснить на каком отрезке временной последовательности лучше проводить обучение НС: на максимально возможном, включаещим в себя разные этапы и состояния рынка, или на отрезке ограниченной длинны, отражающим недавнее актуальное состояние рынка.
Данные по активу за период с 22.05.11 по 23.06.07 (всего 45000 свечей) разбиты на две части: основную и контрольную (40000 и 5000 соответсвенно).
Контрольная часть будет использоваться только для проверки точности предсказаний, обучение НС на ней проводиться не будет.

### **1. Ограниченные отрезки данных**
Основная часть разбита на 6 частей по 15000 свечей со смещением 5000, то есть каждая следующая часть включает в себя 10000 свечей от предыдущей.
Таким образом мы получаем плавную актуализацию состояния рынка. Предстоит обучить и исследовать 6 независимых НС на основе этих частей и проверить точность предсказаний как внутри изучаемой части, так и для контрольной группы, отстоящей по времени от изучаемого участка.

***
**#0029_model (0 - 15000)**

Feed data info:\
mean     60714.797800\
std       2249.392573\
min      51727.000000\
max      71340.000000

TRAINING DATA: Боковое движение после продолжительного медвежьего тренда. Высокая волатильность в начале сэта и низкая в конце.\
TEST DATA: Боковое движение с неудачной попыткой возобновления медвежьего тренда. 

![Feed](image/%230029_1.png) 
![Predictions](image/%230029_2.png) 

#0029_prediction analyzing:\
Mean loss (pt): 52\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 52 | value < -52\
Data compression: 66%\
Filtered matching ratio: 97%

#0029_cont_prediction analyzing:\
Mean loss (pt): 36\
Matching ratio: 94%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 36 | value < -36\
Data compression: 43%\
Filtered matching ratio: 96%

***
**#0030_model (5000 - 20000)**

Data short info:\
mean     60619.766533\
std       1371.421472\
min      51727.000000\
max      64630.000000

TRAINING DATA: боковое движение со средней волатильностью.\
TEST DATA: боковое движение со средней волатильностью.

![Feed](image/%230030_1.png) 
![Predictions](image/%230030_2.png) 

#0030_prediction analyzing:\
Mean loss (pt): 22\
Matching ratio: 89%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 22 | value < -22\
Data compression: 95%\
Filtered matching ratio: 93%

#0030_cont_prediction analyzing:\
Mean loss (pt): 8\
Matching ratio: 94%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 8 | value < -8\
Data compression: 46%
Filtered matching ratio: 97%

***
**#0031_model (10000 - 25000)**

Data short info:\
mean     61727.640333\
std       2731.237092\
min      51727.000000\
max      72427.000000

TRAINING DATA: боковое движение со средней волатильностью.\
TEST DATA: боковое движение переходщее в ярко выраженный бычий тренд.

![Feed](image/%230031_1.png) 
![Predictions](image/%230031_2.png) 

#0031_prediction analyzing:\
Mean loss (pt): 76\
Matching ratio: 87%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 76 | value < -76\
Data compression: 45%\
Filtered matching ratio: 97%

#0031_cont_prediction analyzing:\
Mean loss (pt): 68\
Matching ratio: 89%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 68 | value < -68\
Data compression: 46%\
Filtered matching ratio: 97%

***
**#0032_model (15000 - 30000)**

Data short info:\
mean     65111.634467\
std       4495.326581\
min      58086.000000\
max      72806.000000

TRAINING DATA: боковое движение со средней волатильностью переходящее в ярко выраженный бычий тренд\
TEST DATA: бычий тренд переходящий в боковое движение.

![Feed](image/%230032_1.png) 
![Predictions](image/%230032_2.png) 

#0032_prediction analyzing:\
Mean loss (pt): 29\
Matching ratio: 68%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 29 | value < -29\
Data compression: 75%\
Filtered matching ratio: 71%

#0032_cont_prediction analyzing:\
Mean loss (pt): 21\
Matching ratio: 72%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 21 | value < -21\
Data compression: 72%\
Filtered matching ratio: 82%

***
**#0033_model (20000 - 35000)**
mean     69784.936467
std       4909.039875
min      60594.000000
max      77553.000000

TRAINING DATA: бычий тренд с высокой волатильностью.\
TEST DATA: бычий тренд с низкой волатильностью.

![Feed](image/%230033_1.png) 
![Predictions](image/%230033_2.png) 

#0033_prediction analyzing:\
Mean loss (pt): 43\
Matching ratio: 56%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 43 | value < -43\
Data compression: 74%\
Filtered matching ratio: 57%

#0033_cont_prediction analyzing:\
Mean loss (pt): 35\
Matching ratio: 61%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 35 | value < -35\
Data compression: 77%\
Filtered matching ratio: 60%

***
**#0034_model (25000 - 40000)**

mean     74965.038600\
std       4026.433279\
min      67839.000000\
max      83046.000000

TRAINING DATA: бычий тренд со средней волатильностью.\
TEST DATA: переход к медвежьему тренду.

![Feed](image/%230034_1.png) 
![Predictions](image/%230034_2.png) 

#0034_prediction analyzing:\
Mean loss (pt): 39\
Matching ratio: 62%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 39 | value < -39\
Data compression: 76%\
Filtered matching ratio: 64%

#0034_cont_prediction analyzing:\
Mean loss (pt): 28\
Matching ratio: 63%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 28 | value < -28\
Data compression: 70%\
Filtered matching ratio: 75%

***
В ходе исследований лучшие результаты показала НС 0-15000 (model#0029): 96% - для собственных данных и 94% - для контрольной группы.\
Самый худший результат был у НС 20000-35000 (model#0033): 56% - для собственных данных и 61% для контрольной группы.\
Интересен тот факт, что модель, показавшая лучший результат, была обучена на данных наиболее удаленных по времени от контрольной группы.Анализируя результаты можно заметить, что точность предсказаний корелирует с показателем среднеквадратичного отклонения (STD):\
model#0029 - std 2249,  MR-94%/96%\
model#0030 - std 1371,  MR-89%/94%\
model#0031 - std 2731,  MR-87%/89%\
model#0032 - std 4495,  MR-68%/72%\
model#0033 - std 4909,  MR-56%/61%\
model#0034 - std 4026,  MR-62%/63%\
Чем этот показатель выше, тем ниже точность предсказаний. Таким образом модели обученные на более "спокойных" участках котировок лучше предсказывают направление цены и оказываются более универсальными. Это говорит о том, что, возможно, стоит проводить обучение НС на заранее подготовленных частях данных, отобранных с учетом вышесказанного.

### **2. Обучение на максимально длинном участке данных**
Для обучения взята вся основная часть целиком (40000). Данный блок данных включает в себя различные состояния рынка (флэт, медвежий и бычий тренд) и имеет высокий показатель среднеквадратичного отклонения (7300). Позже, по мере выхода новых данных, возможно как обучение новой НС, так и дообучение старой.

***
**#0036_model (0-25000)**

Data short info:\
count    40000.000000\
mean     66495.028300\
std       7322.697711\
min      51727.000000\
max      83046.000000

![Feed](image/%230036_1.png) 
![Predictions](image/%230036_2.png) 
![Predictions](image/%230036_3.png) 

#0036_prediction analyzing:\
Mean loss (pt): 29\
Matching ratio: 95%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 29 | value < -29\
Data compression: 45%\
Filtered matching ratio: 98%

#0036_contr_part analyzing:
Mean loss (pt): 29
Matching ratio: 95%
Data filtered by column ['Pred_Close_diff']
Condition: value > 29 | value < -29
Data compression: 76%
Filtered matching ratio: 99%

***
НС показала блестящие результаты как на собственном участке, так и на контрольном:
model#0036 -std 7322, MR(test)-95%/98%\ ML(test) -29
                      MR(control)-95%/99%\ ML(control) -29
Стоит отметить, что высокий показатель STD никак не повлиял на точность предсказаний. Из этого следует, что низкие результаты моделей #0031, #0032, #0033, #0034 были связаны с чем-то другим. Возможно со скудостью состояний представленных в анализируемых частях данных.

Далее будет исследован вопрос актуализации модели по мере выхода новых данных. 


### **3. Обучение с дообучением**
Основная часть разбита на стартовую часть (25000) и дополнительные части (по 5000). Первичное обучение НС будет осуществляться на стартовой части, после чего НС будут передаваться дополнительные части для адаптации к изменениям состояния рынка. Точность предсказаний будет замеряться для каждого состояния НС по контрольной группе.

***
**#0035_model (0-25000)**

Data short info:\
count    25000.000000\
mean     61413.022120\
std       2682.369058\
min      51727.000000\
max      72427.000000

![Feed](image/%230035_1.png) 
![Predictions](image/%230035_2.png) 

#0035_prediction analyzing:\
Mean loss (pt): 71\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 71 | value < -71\
Data compression: 33%\
Filtered matching ratio: 99%

#0035_25000-30000_prediction analyzing:\
Mean loss (pt): 29\
Matching ratio: 97%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 29 | value < -29\
Data compression: 51%\
Filtered matching ratio: 99%

#0035_30000-35000_prediction analyzing:\
Mean loss (pt): 29\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 29 | value < -29\
Data compression: 40%\
Filtered matching ratio: 98%

#0035_35000-40000_prediction analyzing:\
Mean loss (pt): 36\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 36 | value < -36\
Data compression: 31%\
Filtered matching ratio: 98%

#0035_control_prediction analyzing:\
Mean loss (pt): 36\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 36 | value < -36\
Data compression: 35%\
Filtered matching ratio: 98%

### Тренировка предобученной модели (model#0035): тонкая настройка (Fine-tune)
***

Дообучение на дополнительных данных:\
range     25000-30000\
fine-tune     (12/32)\
epoch              10\
l_rate       0.000003\
mean     70414.186200\
std       1142.413409\
min      67839.000000\
max      72806.000000


*Вопрос: при обучении программа сохраняет только лучшие результаты по показателю VAL-LOSS. Но в качестве VAL_DATA выступает лишь малая часть дополнительных данных переданных для тренировки. Это может привести к потере универсальности модели. Возможно есть смысл тренировать без VAL_DATA и ориентироваться на TRAINING_LOSS.*


#0035_data_30000-35000+ analyzing:\
Mean loss (pt): 29\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 29 | value < -29\
Data compression: 41%\
Filtered matching ratio: 98%

#0035_data_35000-40000+ analyzing:\
Mean loss (pt): 37\
Matching ratio: 95%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 37 | value < -37\
Data compression: 38%\
Filtered matching ratio: 98%

#0035_contr_part+ analyzing:\
Mean loss (pt): 36\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 36 | value < -36\
Data compression: 40%\
Filtered matching ratio: 98%

***
Дообучение на дополнительных данных:\
range     30000-35000\
fine-tune     (12/32)\
epoch              10\
l_rate       0.000003\
mean     74830.390200\
std       1299.343724\
min      72363.000000\
max      77553.000000

#0035_data_35000-40000++ analyzing:\
Mean loss (pt): 36\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 36 | value < -36\
Data compression: 32%\
Filtered matching ratio: 98%

#0035_contr_part++ analyzing:\
Mean loss (pt): 36\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 36 | value < -36\
Data compression: 37%\
Filtered matching ratio: 98%

***
Дообучение на дополнительных данных:\
range     35000-40000\
fine-tune     (12/32)\
epoch              10\
l_rate       0.000003\
mean     79650.539400\
std       1720.097334\
min      75933.000000\
max      83046.000000

#0035_contr_part+++ analyzing:\
Mean loss (pt): 36\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 36 | value < -36\
Data compression: 40%\
Filtered matching ratio: 98%

***
Дообучение на дополнительных данных:\
range     25000-30000\
fine-tune     (20/32)\
epoch              20\
l_rate       0.000006\
l_rate       0.000003\
mean     70414.186200\
std       1142.413409\
min      67839.000000\
max      72806.000000

#0035_data_30000-35000_01 analyzing:
Mean loss (pt): 29
Matching ratio: 96%
Data filtered by column ['Pred_Close_diff']
Condition: value > 29 | value < -29
Data compression: 37%
Filtered matching ratio: 98%

#0035_contr_part_01 analyzing:
Mean loss (pt): 38
Matching ratio: 95%
Data filtered by column ['Pred_Close_diff']
Condition: value > 38 | value < -38
Data compression: 41%
Filtered matching ratio: 97%

***
Fine-tune (12/32) не дало никаких существенных улучшений, а (20/32) даже немного ухудшило точность. С другой стороны трудно улучшить модель, которая и так имеет точность 98%.