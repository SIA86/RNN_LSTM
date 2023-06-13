# Испытание модели нейронной сети с LSTM слоем
Цель настоящей работы проверить возможность с помощью рекурентной нейронной сети (RNN) на основе предоставленной временной последовательности с данными о цене того или иного биржевого актива выдать более менее точное предсказание (75% или более) цены, которой данный актив достигнет в следующий временной интервал. 
### **1. Влияние типа анализируемых данных (features):**
В ходе исследования предстоит провести испытание моделей с разным исходным набором данных, называемых далее по тексту чертами. Ниже представлен список используемых черт:\
**OHLC** - стандартные ценовые значения актива (цены открытия, макс, мин и цена закрытия за определенный интервал времени (15мин.))\
**Volume** - объем сделок совершенных за опеделенный интервал\
**Arerage(15...50)** - средне арифметическая цена за выбранный период\
**Deviation(15..50)** - среднеквадратичное отклонение цены\
**Corelation(14...32)** - отношение средней цены актива к среднеквадратичному отклонению за выбранный период\
**Local_extrema** - категорийный тип данных, говорящий о том, является ли выбранный интервал локальным минимумом или макмимумом на графике\

Для испытания будут взяты модели со следующими параметрами:

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

***
### **model(#0002)**
Первая модель имеет только одну черту - ['Close']

Predicted data analyzing:\
Mean loss (pt): 113\
Matching ratio: 47%\
Data filtered by column ['Prediction_diff']
Condition: value > 113 | value < -113
Data compression: 74%
Filtered matching ratio: 46%

Средняя ошибка (loss) составила 113 пунктов, однако степень совпадения (Matching ratio) направления движения цены всего 47%. Даже отбрасывание всех предсказаний, значения которых меньше величины средней ощибки, что привелов к компрессии данных в 74%, не дало улучшения параметра 'Matching ratio'

### **model(#0003)**
Модель имеет черты - ['Open','High', 'Low', 'Close']

Mean loss (pt): 96\
Matching ratio: 48%\
Data filtered by column ['Prediction_diff']\
Condition: value > 96 | value < -96\
Data compression: 77%\
Filtered matching ratio: 48%

Добавление черт связанных с параметрами цены актива привело к улучшению показателей. Предсказания следуют за ценой и расположены довольно близко к цене закрытия. Учитывая, что данные параметры нужны для корректного отображения графических результатов вычислений, целесообразно включить эти черты в список обязательных.

![Example](image/%23003_1.png)
![Example](image/%23003_2.png)

### **model(#0004)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Volume']

Predicted data analyzing:\
Mean loss (pt): 105\
Matching ratio: 52%\
Data filtered by column ['Prediction_diff']\
Condition: value > 105 | value < -105\
Data compression: 61%\
Filtered matching ratio: 53%

Добавление черты "объем цены актива" привело к росту средней ошибки, однако немного увеличилась точность предсказаний направления движения цены. На графике видно, что характер линии предсказаний изменился. Часть локальных экстремумов увеличилась в значениях, а часть наоборот уменьшилась. Встречаются существенные расхождения пердсазаний и цены актива. Необходимо ислледовать вклад данной черты в комплексе с другими чертами.

![Example](image/%23004_1.png)
![Example](image/%23004_2.png)

### **model(#0005)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Volume', 'Average15']

Predicted data analyzing:\
Mean loss (pt): 93\
Matching ratio: 48%\
Data filtered by column ['Prediction_diff']\
Condition: value > 93 | value < -93\
Data compression: 72%\
Filtered matching ratio: 45%

Добавление черты "средняя цена актива за период 15" в комплексе с чертой "объем цены актива" привело к незначительному уменьшению средней ошибки, но снизило точность предсказаний направления движения цены. При этом фильтрация результатов только ухудшает точность этого параметра. Характер линии предсказаний также изменился и стал немного ближе к цене актива.

![Example](image/%23005_1.png)
![Example](image/%23005_2.png)

### **model(#0006)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Average15']

Predicted data analyzing:\
Mean loss (pt): 92\
Matching ratio: 50%\
Data filtered by column ['Prediction_diff']\
Condition: value > 92 | value < -92\
Data compression: 66%\
Filtered matching ratio: 51%

Добавление черты "средняя цена актива за период 15" **без учета** черты "объем цены актива" дало результат похожий на model(#0003), но с более точными показателями mean_loss и matching_ratio.

![Example](image/%23006_1.png)
![Example](image/%23006_2.png)

### **model(#0007)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Average15', 'Average25', 'Average50']

Predicted data analyzing:\
Mean loss (pt): 74\
Matching ratio: 48%\
Data filtered by column ['Prediction_diff']\
Condition: value > 74 | value < -74\
Data compression: 77%\
Filtered matching ratio: 48%

Добавление черты "средняя цена актива за период 15, 25, 50" **без учета** черты "объем цены актива" привело к значительному снижению средней ошибки, но снизило точность предсказаний направления движения цены. Фильтрация результатов с компрессией 77% ничего не дала. Линия предсказаний сместилась ниже цены в случае роста актива и выше цены в случае падения. Добавленные черты, являя собой информацию о прошлом, как бы оттягивают линию предсказаний назад и чем этих черт больше добавлено в данные, тем сильнее этот эффект.

![Example](image/%23007_1.png)
![Example](image/%23007_2.png)

### **model(#0008)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Deviation15']

Predicted data analyzing:\
Mean loss (pt): 81\
Matching ratio: 50%\
Data filtered by column ['Prediction_diff']\
Condition: value > 81 | value < -81\
Data compression: 71%\
Filtered matching ratio: 50%


Добавление черты "среднеквадратичное отклонение за период 15" в сравнении с model(#0003) уменьшило среднюю ошибку и немного повысило точность предсказания напрвления движения цены. Фильтрация результатов ничего не дала. Характер линии предсказаний изменился и стал реагировать на изменени параметра STD (например: последовательность свечей с малой волатильностью повысила амплитуду выдаваемых предсказаний).

![Example](image/%23008_1.png)
![Example](image/%23008_2.png)

### **model(#0009)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Deviation15',  'Deviation25',  'Deviation50']

Predicted data analyzing:\
Mean loss (pt): 102\
Matching ratio: 49%\
Data filtered by column ['Prediction_diff']\
Condition: value > 102 | value < -102\
Data compression: 73%\
Filtered matching ratio: 48%

Добавление дополнительных черт  "среднеквадратичное отклонение за период 25 и 50" ухудшило результаты. 

**Испытание model(#0007) и model(#0009) показало, что введение в расчетные данные информации о передыдущих периодах (средняя цена, среднеквадратичное отклонение) способно повлиять на линию предсказаний. При этом эффект усиливается с увеличением периода. Добавление большого количества подобных индикаторов делает предсказания слишком инерционными и снижает точность.**

![Example](image/%23009_1.png)
![Example](image/%23009_2.png)

### **model(#0010)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Volume', 'Average14' 'Deviation14']

Predicted data analyzing:
Mean loss (pt): 121
Matching ratio: 48%
Data filtered by column ['Prediction_diff']
Condition: value > 121 | value < -121
Data compression: 69%
Filtered matching ratio: 47%

Полный отстой

![Example](image/%23010_1.png)
![Example](image/%23010_2.png)

### **model(#0011)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Average14' 'Deviation14']

Predicted data analyzing:
Mean loss (pt): 100
Matching ratio: 49%
Data filtered by column ['Prediction_diff']
Condition: value > 100 | value < -100
Data compression: 71%
Filtered matching ratio: 49%

Полный отстой

![Example](image/%23011_1.png)
![Example](image/%23011_2.png)

### **model(#0012)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Average14' 'Deviation7']

Predicted data analyzing:
Mean loss (pt): 102
Matching ratio: 48%
Data filtered by column ['Pred_Close_diff']
Condition: value > 102 | value < -102
Data compression: 67%
Filtered matching ratio: 47%

Полный отстой

![Example](image/%23012_1.png)
![Example](image/%23012_2.png)

### **model(#0013)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Corelation14']

Predicted data analyzing:
Mean loss (pt): 108
Matching ratio: 50%
Data filtered by column ['Prediction_diff']
Condition: value > 108 | value < -108
Data compression: 66%
Filtered matching ratio: 50%


![Example](image/%23013_1.png)
![Example](image/%23013_2.png)

### **model(#0014)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Local_min_max']

Predicted data analyzing:
Mean loss (pt): 96
Matching ratio: 78%
Data filtered by column ['Pred_Close_diff']
Condition: value > 96 | value < -96
Data compression: 61%
Filtered matching ratio: 86%

Супер!!

![Example](image/%23014_1.png)
![Example](image/%23014_2.png)


### **model(#0015)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Volume', 'Local_min_max']

Predicted data analyzing:
Mean loss (pt): 130
Matching ratio: 62%
Data filtered by column ['Pred_Close_diff']
Condition: value > 130 | value < -130
Data compression: 59%
Filtered matching ratio: 72%

Добавление черты "объем сделок по активу" ухидшило общие показатели относительно результатов model(#0014). Изменился и характер линии предсказаний: появились артефакты схожие с model(#0004). Таким образм, на основании испытаний моделей #0004, #0005, #0010, #0015, можно сделать вывод, что данная черта не является информативной и только добавляет лишний шум к линии предсказаний.

![Example](image/%23015_1.png)
![Example](image/%23015_2.png

### **model(#0016)**
Модель имеет черты - ['Open','High', 'Low', 'Close', 'Corelation14', 'Local_min_max']

Predicted data analyzing:
Mean loss (pt): 101
Matching ratio: 73%
Data filtered by column ['Pred_Close_diff']
Condition: value > 101 | value < -101
Data compression: 63%
Filtered matching ratio: 78%

Добавление черты "Corelation14" - не фантан

![Example](image/%23016_1.png)
![Example](image/%23016_2.png