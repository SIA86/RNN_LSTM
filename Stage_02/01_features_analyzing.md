
### **1. Добавление локальных экстремумов (определение экстремумов происходит внутри отобранной последовательности):**

При добавлении локальных экстремумов к базе данных, подготовленной для обучения НС, необходимо руководствоваться соображением, что для последней свечи в передаваемой последовательности, не может быть установлена категория локального минимума или максимума. Для этого нужны сведения о цене закрытия следующей свечи, которой на момент анализа у нас нет. Таким образом последняя свеча в последовательности будет всегда иметь категорию 0 (то есть отсутствие экстремума).

![Data](image/data.png)

Для испытания будут взяты модели со следующими параметрами:

Data options:\
LENGTH = 10000\
ROUND = 5\
NORM = 'std'

Dataset options:\
LABEL_SIZE = 1\
SHIFT = 1\
WARM_UP = 100\
LABEL_NAMES = ['Close']

RNN options:\
NEURONS = 8\
L_RATE = 0.00003\
LOSS = 'mean_absolute_error'\
METR = 'mean_absolute_error'

***
### **model(#0001)**
FEATURES: ['Open','High', 'Low', 'Close', 'Loc_Min', Loc_Max]\
INP_SIZE = 6 

![Prediction](image/%230001_closeup.png)

#0001_prediction analyzing:\
Mean loss (pt): 43\
Matching PC ratio: 51%\
Matching PP ratio: 51%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -43 | value > 43\
Data compression: 76%\
Filtered matching PC ratio: 53%\
Filtered matching PP ratio: 49%

***
### **model(#0002)**
FEATURES: ['Open','High', 'Low', 'Close', 'Loc_Min', Loc_Max]\
INP_SIZE = 12\

![Prediction](image/%230002_closeup.png)

#0002_prediction analyzing:\
Mean loss (pt): 44\
Matching PC ratio: 50%\
Matching PP ratio: 51%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -44 | value > 44\
Data compression: 75%\
Filtered matching PC ratio: 55%\
Filtered matching PP ratio: 51%

***
### **model(#0003)**
FEATURES: ['Open','High', 'Low', 'Close', 'Loc_Min', Loc_Max]\
INP_SIZE = 18\

![Prediction](image/%230003_closeup.png)

#0003_prediction analyzing:\
Mean loss (pt): 45\
Matching PC ratio: 51%\
Matching PP ratio: 51%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -45 | value > 45\
Data compression: 75%\
Filtered matching PC ratio: 55%\
Filtered matching PP ratio: 50%

***
### **model(#0004)**
FEATURES: ['Open','High', 'Low', 'Close', 'Loc_Min', Loc_Max]\
INP_SIZE = 24\

![Prediction](image/%230004_closeup.png)

#0004_prediction analyzing:\
Mean loss (pt): 45\
Matching PC ratio: 51%\
Matching PP ratio: 52%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -45 | value > 45\
Data compression: 74%\
Filtered matching PC ratio: 54%\
Filtered matching PP ratio: 50%

***
### **model(#0005)**
FEATURES: ['Open','High', 'Low', 'Close', 'Corelation6' 'Loc_Min', Loc_Max]\
INP_SIZE = 6\

![Prediction](image/%230005_closeup.png)

#0005_prediction analyzing:\
Mean loss (pt): 44\
Matching PC ratio: 51%\
Matching PP ratio: 51%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -44 | value > 44\
Data compression: 76%\
Filtered matching PC ratio: 51%\
Filtered matching PP ratio: 50%

***
### **model(#0006)**
FEATURES: ['Open','High', 'Low', 'Close', 'Corelation6' 'Loc_Min', Loc_Max]\
INP_SIZE = 15\
SHIFT = 5 - predict 5th candle after sequence

![Prediction](image/%230006_closeup.png)

#0006_prediction analyzing:\
Mean loss (pt): 89\
Matching PC ratio: 54%\
Matching PP ratio: 48%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -89 | value > 89\
Data compression: 77%\
Filtered matching PC ratio: 58%\
Filtered matching PP ratio: 48%

***
### **model(#0007)**
FEATURES: ['Open','High', 'Low', 'Close', 'Loc_Min', Loc_Max]\
INP_SIZE = 10\
SHIFT = 1\
L_RATE = 0.0001\
LOSS = 'mean_squared_logarithmic_error'\
EPOCH = 50

![Prediction](image/%230007_closeup.png)

#0007_prediction analyzing:\
Mean loss (pt): 73\
Matching PC ratio: 54%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -73 | value > 73\
Data compression: 71%\
Filtered matching PC ratio: 53%

***
### **model(#0008)**
FEATURES: ['Open','High', 'Low', 'Close', 'Loc_Min', Loc_Max]\
INP_SIZE = 10\
SHIFT = 1\
L_RATE = 0.0001\
LOSS = 'mean_squared_error'\
EPOCH = 50

![Prediction](image/%230008_closeup.png)

#0008_prediction analyzing:\
Mean loss (pt): 45\
Matching PC ratio: 50%\
Data filtered by column ['Pred_Close_diff']\
Condition: value < -45 | value > 45\
Data compression: 75%\
Filtered matching PC ratio: 53%

В сравнении с model(#0002) никаких существенных отличий изменение loss function не дало.

***
Точность пердсказаний всех моделей не превышает 54%, что практически совпадает с точностью предсказаний без передачи НС сведений о локальных экстремумах.