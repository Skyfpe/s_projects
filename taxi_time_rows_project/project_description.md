#  Прогнозирование заказов такси

Такси-компания собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Постройте модель для такого предсказания.

Задача: **Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.**

**План работы**:

1. Загрузить данные и выполнить их ресемплирование по одному часу.
2. Проанализировать данные.
3. Обучить разные модели с различными гиперпараметрами. Сделать тестовую выборку размером 10% от исходных данных.
4. Проверить данные на тестовой выборке и сформулировать вывод.


Данные лежат в файле `taxi.csv`. Количество заказов находится в столбце `num_orders`.

<h1>Содержание<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Загрузка-данных" data-toc-modified-id="Загрузка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Загрузка данных</a></span><ul class="toc-item"><li><span><a href="#Загружаем-данные-и-смотрим,-с-чем-нам-придется-работать" data-toc-modified-id="Загружаем-данные-и-смотрим,-с-чем-нам-придется-работать-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Загружаем данные и смотрим, с чем нам придется работать</a></span></li><li><span><a href="#Смотрим-графики" data-toc-modified-id="Смотрим-графики-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Смотрим графики</a></span></li></ul></li><li><span><a href="#Анализ" data-toc-modified-id="Анализ-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ</a></span><ul class="toc-item"><li><span><a href="#Разложение-на-компоненты" data-toc-modified-id="Разложение-на-компоненты-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Разложение на компоненты</a></span></li></ul></li><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#Классические-модели" data-toc-modified-id="Классические-модели-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Классические модели</a></span></li><li><span><a href="#LightGBM" data-toc-modified-id="LightGBM-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>LightGBM</a></span></li><li><span><a href="#Catboost" data-toc-modified-id="Catboost-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Catboost</a></span></li></ul></li><li><span><a href="#Тестирование" data-toc-modified-id="Тестирование-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Тестирование</a></span></li><li><span><a href="#Вывод и рекомендации заказчику" data-toc-modified-id="Вывод и рекомендации заказчику-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Вывод и рекомендации заказчику</a></span></li></ul></div>
