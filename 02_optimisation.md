# Оптимизация

Оптимизация модели с лучшими показателями совпадения направления движения цены (Matching).\
Модель - model(#0014)

### Модель имеет следующимие параметры:

FEATURES = ['Open','High', 'Low', 'Close', 'Local_min_max']

Data options:\
LENGTH = last 12000\
ROUND = 6\
NORM = 'std'

Dataset options:\
INP_SIZE = 15 #length of candles sequence to analize\
LABEL_SIZE = 1\
SHIFT = 1\
LABEL_NAMES = ['Close']

RNN options:\
NEURONS = 8\
L_RATE = 0.001\
LOSS = 'mean_absolute_error'\
METR = 'mean_absolute_error'

## Испытания
***
### **1. Проверка влияния длинны последовательности, передаваемой нейронной сети для анализа**

**Model(#0014) - контрольная**\
INPUT_SIZE = 15

Predicted data analyzing:\
Mean loss (pt): 96\
Matching ratio: 78%\
After filtration:\
Data compression: 61%\
Filtered matching ratio: 86%

**Model(#0017)**\
INPUT_SIZE = 25

Predicted data analyzing:\
Mean loss (pt): 93\
Matching ratio: 77%\
After filtration:\
Data compression: 60%\
Filtered matching ratio: 76%

**model(#0018)**\
INPUT_SIZE = 5

Predicted data analyzing:\
Mean loss (pt): 58\
Matching ratio: 89%\
After filtration:\
Data compression: 48%\
Filtered matching ratio: 96%

**model(#0019)**\
INPUT_SIZE = 10

Predicted data analyzing:\
Mean loss (pt): 88\
Matching ratio: 87%\
After filtration:\
Data compression: 48%\
Filtered matching ratio: 93%

Проверка моделей #0017, #0018, #0019 показала, что увеличение длинны последовательности вовсе не приводит к улучшению результатов. Напротив последрвательности меньшей длинны оказались более точными. Так с параметром INPUT_SIZE = 5 точность составила 89% и 96% с фильтрацией.

### **2. Проверка влияния количества NEURONS, L_RATE и EPOCHS на точность пердсказаний**

**model(#0018) - контрольная**\
NEURONS = 8\
L_RATE = 0.001\
EPOCHS = 15

TRAINING_ACC = 0.1794\
VAL_ACC = 0.0315\
TEST_ACC = 0.0264

Predicted data analyzing:\
Mean loss (pt): 58\
Matching ratio: 89%\
After filtration:\
Data compression: 48%\
Filtered matching ratio: 96%

Время вычисления 11 сек один EPOCH

**model(#0020)**\
NEURONS = 32\
L_RATE = 0.001\
EPOCHS = 15

TRAINING_ACC = 0.1244\
VAL_ACC = 0.03556\
TEST_ACC = 0.0374

Predicted data analyzing:\
Mean loss (pt): 82\
Matching ratio: 74%\
After filtration:\
Data compression: 54%\
Filtered matching ratio: 85%

Увеличение числа нейронов уменьшило ошибку TRAINIG_ACC, но увеличило ошибку VAL_ACC и TEST_ACC. Точность передсказаний снизилась. Время вычисления увеличилось в 10-12 раз (133 сек один EPOCH). Вероятно дело в слишком высоком L_RATE. При градиентном спуске мы просто "пролетаем" минимум.

**model(#0021)**\
NEURONS = 16\
L_RATE = 0.001\
EPOCHS = 15

TRAINING_ACC = 0.1357\
VAL_ACC = 0.0401\
TEST_ACC = 0.0421

Predicted data analyzing:\
Mean loss (pt): 93\
Matching ratio: 85%\
After filtration:\
Data compression: 51%\
Filtered matching ratio: 94%

Точность чуть ниже, чем в model(#0018). Время вычисления относительно контрольной модели увеличилось в 3 раза (33 сек. один EPOCH)

**model(#0022)**\
NEURONS = 4\
L_RATE = 0.001\
EPOCHS = 15

TRAINING_ACC = 0.2199\
VAL_ACC = 0.04949\
TEST_ACC = 0.0618

Predicted data analyzing:\
Mean loss (pt): 136\
Matching ratio: 73%\
After filtration:\
Data compression: 58%\
Filtered matching ratio: 77%

Время вычисления относительно контрольной модели уменьшилось в 1.5 раза (8 сек. один EPOCH). Ошибка ощутимо увеличилась, точность снизилась.

**model(#0023)**\
NEURONS = 8 \
LEARNING_RATE = 0.0001
EPOCHS = 50

TRAINING_ACC = 0.1653\
VAL_ACC = 0.0284\
TEST_ACC = 0.0278

Predicted data analyzing:
Mean loss (pt): 61
Matching ratio: 91%
After filtration:\
Data compression: 46%
Filtered matching ratio: 97%

**model(#0024)** \
NEURONS = 48\
LEARNING_RATE = 0.0001
EPOCHS = 100

TRAINING_ACC = 0.0920\
VAL_ACC = 0.0267\
TEST_ACC = 0.0229

Predicted data analyzing:
Mean loss (pt): 50
Matching ratio: 93%
After filtration:\
Data compression: 44%
Filtered matching ratio: 98%

Время вычислений 240 сек. один EPOCH (240*100= 6.5 часов). Лучший результат VAL_ACCURACY был достигнут на 16 EPOCH. Начиная с 30-35 EPOCH TRAINING_LOSS достигла своего минимума и далее колебалась рядом с этим значением. Возможно не хватило L_RATE чтобы выскочить из локального минимума. Это можно проверить слегда изменив L_RATE.

**model(#0025)** \
NEURONS = 48\
LEARNING_RATE = 0.00005\
EPOCHS = 100

TRAINING_ACC = 0.0902\
VAL_ACC = 0.02817\
TEST_ACC = 0.0249

Predicted data analyzing:\
Mean loss (pt): 55\
Matching ratio: 96%\
After filtration:\
Data compression: 35%\
Filtered matching ratio: 98%

![Loss_chart](image/%230025_1.png)

**model(#0026)** \
NEURONS = 16\
LEARNING_RATE = 0.00001
EPOCHS = 500

TRAINING_ACC = 0.1206\
VAL_ACC = 0.02848\
TEST_ACC = 0.0269

Predicted data analyzing:\
Mean loss (pt): 59       \
Matching ratio: 91%\
After filtration:\
Data compression: 45%\
Filtered matching ratio: 97%

![Loss_chart](image/%230026_1.png)


**model(#0027)** \
NEURONS = 64\
LEARNING_RATE = 0.00001\
EPOCHS = 25

TRAINING_ACC = 0.0907\
VAL_ACC = 0.03165\
TEST_ACC = 0.0293

Predicted data analyzing:\
Mean loss (pt): 64\
Matching ratio: 78%\
After filtration:\
Data compression: 53%\
Filtered matching ratio: 93%

![Loss_chart](image/%230027_1.png) 

**model(#0028)** \
NEURONS = 48\
LEARNING_RATE = 0.00003\
EPOCHS = 50

Predicted data analyzing:\
Mean loss (pt): 51\
Matching ratio: 97%\
After filtration:\
Data compression: 31%\
Filtered matching ratio: 99%

TRAINING_ACC = 0.0866\
VAL_ACC = 0.02604\
TEST_ACC = 0.0232


![Loss_chart](image/%230028_1.png) 