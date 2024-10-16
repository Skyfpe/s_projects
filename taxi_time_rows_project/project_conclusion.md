## Вывод и рекомендации заказчику

**Итоговый вывод:** выбранная модель показала неплохой результат (что также подтверждается графиками остатков), удовлетворяющий условиям, и готова к работе.

Также в дальнейшем можно попробовать улучшить модель:
1. Например, подобрать нужные гиперпараметры (что под вопросом - мы пытались и любые настройки стабильно делали работу моделей хуже, как на тестовых, так и на обучающих данных; вероятно, это связано с относительной простотой имеющихся временных рядов и отсутствием какх-то чересчур сложных закономерностей);
2. Поработать с данными. В данных есть сильные шумы, и не особо понятно что это. Это могут быть совершенно случайные флуктуации или неучтенные праздники/акции по тарифам в компании/крупные события/etc. Наличие данных о таких праздниках, а также данных за начало текущего и аналогичный период предыдущего года, вероятно, могли бы внести некоторую ясность в природу наблюдаемых шумов и в понимание того, почему к концу периода их стало относительно много (больше, чем в начале года).
3. Поработать с признаками. Все признаки (кроме года) оказались плюс-минус вносящими свою лепту, и можно было бы попробовать пообучать различные модели на разном количестве lag'ов (в разумных пределах, конечно), пока они не перестанут вносить вклад в предсказание. Также можно поиграться и подобрать другое значение скользящего среднего.
