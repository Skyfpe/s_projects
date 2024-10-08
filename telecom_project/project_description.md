# Проект: телекоммуникации

**Описание**

Оператор связи намерен бороться с оттоком клиентов. Для этого его руководство хочет, чтобы его сотрудники начали предлагать промокоды и специальные условия всем, кто планирует отказаться от услуг связи. Чтобы заранее выявлять таких пользователей, заказчику нужна модель, которая будет предсказывать, разорвёт ли абонент договор.

Команда оператора собрала персональные данные о некоторых клиентах, информацию об их тарифах и услугах. Наша задача — обучить на этих данных модель для прогноза оттока клиентов.

**План работы**

1. Загрузить и осмотреть данные
2. Выполнить предобработку:
    - привести данные к соответствующим типам
    - стандартизировать наименования столбцов и текстовые категориальные данные
    - удалить дубликаты
    - заполнить пропуски
3. Повести исследовательский анализ данных:
    - Провести анализ количественных признаков
    - Провести анализ временных признаков - дат
    - Провести анализ категориальных признаков
4. Провести корреляционный анализ:
    - На основе имеющихся линейных зависимостей в данных оценить, какие модели следует использовать
    - Оценить, какие признаки следует оставить, а какие явно можно удалить
5. Подготовить данные:
    - Удалить ненужные признаки
    - Выделить тренировочную и тестовую выборки
    - Закодировать и отмасштабировать данные
6. Обучить модели и найти лучшую, которую впоследствии будем тестировать:
    - применить кросс-валидацию для поиска моделей
    - проверить модели на адекватность
    - выбрать лучшую модель, которая подвергнется тестированию
7. Протестировать лучшую найденную модель:
    - оценить её качество на тестовой выборке
    - оценить вклад признаков в классификацию
    - построить матрицу ошибок модели
    - выделить рекомендации по дальнейшему улучшению и работе итоговой модели
8. Написать итоговый вывод

<h1>Содержание<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Шаг.-Загрузка-данных" data-toc-modified-id="Шаг.-Загрузка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Шаг. Загрузка данных</a></span></li><li><span><a href="#шаг.-Предобработка-данных" data-toc-modified-id="шаг.-Предобработка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>шаг. Предобработка данных</a></span></li><li><span><a href="#шаг.-Исследовательский-анализ" data-toc-modified-id="шаг.-Исследовательский-анализ-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>шаг. Исследовательский анализ</a></span><ul class="toc-item"><li><span><a href="#Анализ-количественных-и-временных-и-дат" data-toc-modified-id="Анализ-количественных-и-временных-и-дат-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Анализ количественных и временных и дат</a></span></li><li><span><a href="#Анализ-категориальных-признаков" data-toc-modified-id="Анализ-категориальных-признаков-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Анализ категориальных признаков</a></span></li><li><span><a href="#Вывод-по-разделу" data-toc-modified-id="Вывод-по-разделу-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Вывод по разделу</a></span></li></ul></li><li><span><a href="#шаг.-Корреляционный-анализ" data-toc-modified-id="шаг.-Корреляционный-анализ-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>шаг. Корреляционный анализ</a></span></li><li><span><a href="#шаг.-Подготовка-данных" data-toc-modified-id="шаг.-Подготовка-данных-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>шаг. Подготовка данных</a></span></li><li><span><a href="#шаг.-Обучение-моделей" data-toc-modified-id="шаг.-Обучение-моделей-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>шаг. Обучение моделей</a></span><ul class="toc-item"><li><span><a href="#Дерево-решений,-случайный-лес" data-toc-modified-id="Дерево-решений,-случайный-лес-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Дерево решений, случайный лес</a></span></li><li><span><a href="#LightGBM" data-toc-modified-id="LightGBM-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>LightGBM</a></span></li><li><span><a href="#CatBoost" data-toc-modified-id="CatBoost-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>CatBoost</a></span></li><li><span><a href="#Проверка-на-адекватность" data-toc-modified-id="Проверка-на-адекватность-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Проверка на адекватность</a></span></li><li><span><a href="#Выбор-лучшей-модели" data-toc-modified-id="Выбор-лучшей-модели-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Выбор лучшей модели</a></span></li></ul></li><li><span><a href="#шаг.-Анализ-лучшей-модели" data-toc-modified-id="шаг.-Анализ-лучшей-модели-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>шаг. Анализ лучшей модели</a></span></li><li><span><a href="#Итоговый-отчет" data-toc-modified-id="Итоговый-отчет-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Итоговый отчет</a></span></li></ul></div>
