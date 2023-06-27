# Исследование способности НС обученной на одном таймфрейме делать точные предсказания на другом
Испытания проводятся на model(#0036), которая была обучена на пятиминутном таймфрейме.

### **1. Таймфрэем 15 мин.**

#0036_SPFB.Si_220511_230607_15min analyzing:\
Mean loss (pt): 96\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 96 | value < -96\
Data compression: 49%\
Filtered matching ratio: 99%

![Predictions](image/%230036_15мин.png) 
![Predictions](image/%230036_15мин-1.png) 
![Predictions](image/%230036_15мин-2.png) 

### **2. Таймфрэйм 1 час.**

#0036_SPFB.Si_220511_230607_1h analyzing:\
Mean loss (pt): 146\
Matching ratio: 96%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 146 | value < -146\
Data compression: 72%\
Filtered matching ratio: 99%

![Predictions](image/%230036_1час.png) 
![Predictions](image/%230036_1час-1.png) 
![Predictions](image/%230036_1час-2.png) 

### **2. Таймфрэйм 1 день.**

#0036_SPFB.Si_220511_230607_1d analyzing:\
Mean loss (pt): 778\
Matching ratio: 71%\
Data filtered by column ['Pred_Close_diff']\
Condition: value > 778 | value < -778\
Data compression: 99%\
Filtered matching ratio: 60%

![Predictions](image/%230036_1день.png) 
![Predictions](image/%230036_1день-1.png) 
![Predictions](image/%230036_1час-2.png)

***
Исследование показало, что НС, обученная на пятиминутном таймфрейме выдает предсказания без потери точности для пятнадцатимунтного и часового интервалов, однако не может справиться с дневным. Также с изменением таймфрейма выростает средняя ошибка.

5 мин. L: 88, M: 95%\
15 мин L: 96, M: 96%\
1 час L: 146, M: 96%\
1 день L: 778, M: 71%