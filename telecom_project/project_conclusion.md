## Итоговый отчет

**Проект завершен, основное условие выполнено (ROC-AUC на тестовых данных >= 0.85). Вот ход работы и проделанные манипуляции согласно плану:**
1. **Загрузка данных**
    - Данные о клиентах содержались в четырёх разных датасетах.
    - При осмотре явных аномалий не было выявлено.
    - В первом датасете было 11 пропусков в колонке 'total_charges', мы их удалили. 
    - Некоторые данные не были приведены к соответствующим типам.
    - Для удобства работы все датасеты были объединены по ключу - 'client-id'.
    - после объединения образовались пропуски.
    
2. **Предобработка данных**
    - Была проведена стандартизация категориальных данных и наименований столбцов.
    - Был произведен поиск и удаление явных дубликатов.
    - Был произведен поиск и удаление неявных дубликатов.
    - Были заполнены пропуски, которые наблюдались только в категориальных колонках, значением 'undefined'.
    - Так как в данных не было явного целевого признака для предсказания, мы выделили его из колонок 'begin_date' и 'end_date'.
    - Ушедшие клиенты были помечены как '1', оставшиеся - как '0'.
    
3. **Исследовательский анализ**
    - Был проведен исследовательский анализ количественных признаков:
        - Первичные выводы:
            1. Есть выбросы в колонке 'total_charges', однако  вряд ли можно считать их аномалиями: значения находятся в разумных пределах.
            2. Распределение сумм месячных платежей бимодальное с двумя пиками в районе 20 и ±80. Наиболее вероятно, что это связано с особенностями тарифов и пакетами предоставляемых услуг, отсюда клиентов можно поделить на две условные группы: пользующихся базовыми тарифами и пакетами и пользующихся тарифами и пакетами "плюс".
            3. В среднем чаще уходят клиенты, платящие по тарифам "плюс". Это плохой сигнал для бизнеса: ему действительно нужно работать над удержанием т.н. "китов".
            4. По общим суммам платежей примерно та же картина: в среднем ушедшие платят ощутимо больше, чем оставшиеся.
        - Портрет собравшегося уходить клиента (согласно анализу количественных признаков и дат):
            1. В среднем договор расторгают наиболее давние клиенты.
            2. По сумме платежей есть различия: обычно расторгающие договор клиенты являются давними и платят в общем небольшую (среднюю) сумму, однако их месячные платежи довольно высоки. Это говорит о том, что данные клиенты расторгли договор довольно быстро, хоть и заключили его довольно давно, поэтому как таковыми 'давними' они не являются. Исходя из этой находки, сформируем новый признак 'длительность', который покажет время пребывания клиента с компанией, и построим для него те же графики.
        - Окончательные выводы:
            1. Уходят в основном клиенты со средней длительностью действия контракта. "Старожилы" и "новички", как правило, продолжают пользоваться услугами компании: среди наиболее долго взаимодействующих с компанией клиентов (порядка 2000 дней) практически нет тех, кто расторг договор, а среди новичков есть небольшой процент ушедших, которым, похоже, не подошли условия компании. Пик расторжения приходится на 1000 дней (примерно 2.73 года), и, похоже, это действительно как-то связано с изменением (или, наоборот, с отсутствием изменений) в перечне оказываемых услуг и тарифов или их цене. Уполномоченному отделу стоит обратить внимание на этот факт.
            2. Ушедшие клиенты в среднем за весь срок действия договора платили плюс-минус среднюю цену (в районе 3000). Это соотносится с тем, что мы видели.
            3. Ушедшие клиенты, как правило, платили довольно много помесячно, то есть скорее всего пользовались тарифами категории 'плюс'. Из этого вытекают два вывода:
                - Предварительно: причина расторжения договора, вероятно, заключается не в изменении цены, а в изменении качества и/или количества услуг самой компании или конкурентов;
                - Если причина кроется не в качестве или перечне предоставляемых услуг, то проблема стоит внимания аналитиков. В любом случае, на этот факт следует обратить внимание уполномоченному отделу.
                
   - Был проведен анализ категориальных признаков.
       - Вывод:
           - Распределения клиентов относительно таргета по признакам, связанным с предоставляемыми компанией услугами, наталкивает на определенные мысли. Похоже, уходящие клиенты по какой-то причине недовольны услугами провайдера. Непонятно, связано это непосредственно с качеством услуг или с услугами, предлагаемыми конкурентами. Тема требует более детального разбирательства. Также количество и распределение клиентов по таргету в заполненных пропусках в тех же признаках остается неизменным. Скорее всего, противореча ранее обозначенному предположению, на уход пользователя не влияет наличие или отсутствие той или иной услуги, проблема кроется в самой услуге непосредственно.
           
   - Был проведен анализ зависимости длительности действия контракта и перечня предоставляемых услуг:
       - Вывод:
           - Видимая зависимость между известным и неизвестным типом услуг и длительностью, а также расторжением контракта безусловно есть, однако её природа остается тайной, покрытой мраком. Видна определенная историческая закономерность, возможно, в прошлом у компании происходило что-то такое, что побуждало клиентов чаще расторгать договор. Также неясно, неизвестные услуги являются проблемой выгрузки или самих данных непосредственно; данная проблема требует более детального изучения аналитиками компании. Тем не менее, даже с учетом недостатка информации, открытая закономерность должна положительно сказаться на моделировании и помочь предупреждать подобные тренды в будущем.
           
   - **Общий вывод по разделу:**
        1. От компании уходят преимущественно те, кто платит по договору выше среднего в месяц. С этим действительно надо что-то делать.
        2. Скорее всего отток клиентов связан непосредственно с услугами компании, однако неясно, связан он с их качеством или условиями рынка. Требуется более масштабный и детальный анализ. Признаки с услугами, скорее всего, при моделировании будут учтены как определяющие.
        3. Уходят преимущественно наиболее старые клиенты с ожидаемой средней длительностью действия контракта в ±1000 дней.
        4. Прослеживается тренд оттока; учитывая пункт выше, он носит предположительно исторический характер, а также как-то связан с состоянием компании или исторической рыночной конъюнктурой.
        5. Есть относительные различия в оттоке пользователей с установленными и неустановленными услугами. Неустановленные услуги (пропуски) могут представлять маску других услуг или уже имеющихся с разным распределением по категориям. В любом случае, имеющейся информации должно оказаться достаточно для моделирования, однако было бы неплохо в дальнейшем уточнить категории и/или избегать ошибок при выгрузке.

4. **Корреляционный анализ признаков**
    - Были удалены признаки 'begin_date' и 'end_date' - первый является бесполезным в принципе, а второй привел бы к утечке целевого признака.
    - Был произведен корреляционный анализ признаков с помощью линейного коэффициента корреляции Пирсона:
        - Выводы:
            1. Признак 'gender' вообще практически никак не коррелирует с другими признаками. Исключения составляют таргет и ежемесячная сумма платежей. Признак удалять не будем, так как он может оказаться полезным в качестве уточняющего.
            2. Видим жёлтый "островок" или кластер предоставляемых услуг в центре графика. коэффициент линейной корреляции между собой они держат высокий (>0.9), однако их удаление под большим вопросом: все они по-разному коррелируют с таргетом, поэтому их совокупность и, вероятно, даже каждый из них в итоге может оказаться полезным. Линейные алгоритмы в такой ситуации использовать имеет мало смысла, поэтому стоит попробовать нелинейные, в особенности деревья решений и бустинги.
            3. Все услуги крепко связаны с 'monthly_charges'. Это логично, ведь от набора услуг (пакета) зависит и месячная плата.
            4. Единственный коэффициент корреляции >0.9 вне кластера - между признаками 'internet_service' и 'monthly_charges'. Та же история, скорее всего 'internet_service' будет уточняющим при моделировании.
5. **Подготовка данных**
    - Было произведено разделение на тренировочную и тестовую выборки.
    - Поскольку было решено не использовать линейные модели, было произведено кодирование входных признаков посредством OrdinalEncoder.
    - Ранее заполненные пропуски были перекодированы, чтобы у них были одинаковые значения.
    - Из-за малого разнообразия данных и сильного дисбаланса классов было применено сэмплирование тренировочной выборки посредством метода SMOTETomek.
    - Несколько методов масштабирования количественных данных было добавлено в пайплайн.
6. **Обучение моделей**
    - При обучении (и тестировании) мы ориентируемся на метрику ROC-AUC
    - При обучении использовался поиск посредством кросс-валидации. Всего в обучении участвовало четыре алгоритма:
        - Дерево решений:
            - Метрика лучшей модели при кросс-валидации: 0.931
            - Параметры лучшей модели:
                - 'prep__num': RobustScaler(),
                - 'model__max_depth': 6,
                - 'model__min_samples_leaf': 2,
                - 'model__max_features': 16,
        - Случайный лес:
            - Метрика лучшей модели при кросс-валидации: 0.828
            - Параметры лучшей модели:
                - 'prep__num': MinMaxScaler(),
                - 'model__n_estimators': 42,
                - 'model__min_samples_leaf': 3,
                - 'model__max_features': 7,
                - 'model__max_depth': 6,
        - LightGBM:
             - Метрика лучшей модели при кросс-валидации: 0.876
             - Параметры лучшей модели:
                 - 'prep__num': RobustScaler(),
                 - 'model__learning_rate': 0.5,
                 - 'model__max_depth': 3,
                 - 'model__metric': 'AUC',
        - CatBoost:
             - Метрика лучшей модели при кросс-валидации: 0.892
             - Параметры лучшей модели:
                 - 'prep__num': RobustScaler(),
                 - 'model__cat_features': None,
                 - 'model__eval_metric': 'AUC',
                 - 'model__iterations': 5000,
                 - 'model__learning_rate': 0.1,
                 - 'model__loss_function': 'Logloss',
                 - 'model__od_type': 'Iter',
                 - 'model__od_wait': 50,
                 - 'model__thread_count': -1,
                 
    - Были построены графики ROC-кривых для предсказаний моделей с найденными параметрами на тренировочной выборке чтобы оценить обобщающую способность каждой из них.
    - По итогу лучшей моделью была выбрана модель CatBoost.
7. **Анализ лучшей модели**
    - ROC-AUC выбранной CatBoost-модели на тестовой выборке составил 0.898
    - Была произведена оценка важности признаков:
        - Все количественные признаки попали в топ-3. Это косвенно подтверждает предположение о малом разнообразии данных. Самый полезный признак - 'duration_days', его мы выделили сами. Почти все категориальные, так называемые "услуговые" признаки оказались плюс-минус средней полезности, как мы и предположили в разделе с корреляционным анализом. Это же относится к признаку 'gender'. Самым малополезным оказался признак 'internet_service'. Все признаки так или иначе участвовали в предсказании.
    - Была построена матрица ошибок модели на тестовой выборке.
    - Матрица ошибок показала, что у модели низкий показатель recall - а именно он нужен заказчику для определения потенциальных "уходильщиков":
        - По идее, нестрашно, если некоторой доле правильно определенных остающихся клиентов будут предложены промокоды и спец. условия, а вот пропуск и неправильное определение потенциально уходящих клиентов не позволит эффективно применять задуманные мероприятия по их удержанию, учитывая, что именно эта категория в среднем более платежеспособна.
    - Было произведено мини-исследование с подбором порогов классификации, которое показало, что:
        1. Похоже, что даже при очень низком пороге классификации количество правильно предсказанных объектов класса '0' держится на приемлемом уровне. По всей видимости наша модель очень уверена в предсказании объектов с классом '0', то есть тех, кто не собирается уходить, чего не скажешь об объектах класса '1'.
        2. При пороге классификации 0.1 правильно будут определены 197 потенциально уходящих клиентов из 275, и всего 131 неправильно определенный клиент, который уходить не собирался, из 1482. Это вполне хороший баланс, который можно считать приемлемым.
        3. При пороге классификации 0.01 правильно будут определены аж 240 потенциально уходящих клиентов из 275, и уже 407 неправильно определенный клиент, который уходить не собирался, из 1482. Здесь баланс уже сильно смещен: модель становится несколько капризной при таком пороге и почти не ошибается в классе '1'. Наверное, это удовлетворительный расклад.
        4. Это предварительные варианты порогов классификации, которые, тем не менее, можно считать рабочими. Точная "пороговая донастройка" требует более детального анализа клиентских платежей, прибылей и рисков и зависит от нацеленных на удержание клиентов кампаний, а также финансовых бюджетов и финансовых рисков этих кампаний: необходима консультация с отвественными лицами/отделом компании-заказчика и доп. исследование. Также можно ещё отмасштабировать пороги классификации и настроить их в разряде тысячных и даже десятитысячных.

**Рекомендации заказчику:**
- Желательно провести дополнительное исследование по поводу качества и количества услуг компании, а также исторических данных о рыночной конъюнктуре.
- Заданное условие достигнуто (ROC-AUC на тестовых данных >= 0.85), построенная модель функционирует и готова к работе. Были предложены рабочие варианты порогов классификации для построенной модели, при которых она гораздо успешнее определяет потенциальных клиентов, которые могут расторгнуть договор, и при которых количество неправильно определенных клиентов, не собирающихся уходить от компании, находится на приемлемом уровне. Для более тонкой настройки требуется консультация с представителями компании и доп. аналитическое исследование.
