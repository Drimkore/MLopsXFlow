# MLopsXFlow
Беззубов Артем Александрович Группа РИМ-220907  
### Эксперименты
Для конвеера использовался airflow.  
Эксперимета проводились при помощи mlflow.  
Осуществлял подбор гиперпараметров. 
При тренировке модели (эксперимент training_mod) прогоняются различные варианты гиперпараметров отдельными запусками эксперимента. Затем идет запуск, в котором считываются результаты запусков с подбором гиперпараметров, и наилучшие результаты используются для итоговой тренировки модели.
После, в следующем эксперименте, смотрится результат на тестовой выборке.


### Airflow
![](https://github.com/Drimkore/MLopsXFlow/blob/main/screens/af.jpg)
### Mlflow
![](https://github.com/Drimkore/MLopsXFlow/blob/main/screens/trainmod.jpg)
![](https://github.com/Drimkore/MLopsXFlow/blob/main/screens/testingmod.jpg)
![](https://github.com/Drimkore/MLopsXFlow/blob/main/screens/trainrun.jpg)
![](https://github.com/Drimkore/MLopsXFlow/blob/main/screens/testrun.jpg)
