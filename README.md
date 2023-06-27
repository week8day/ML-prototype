![Logo](/media/logo.png)
# Кредитный скоринг клиентов банка

Задача состоит в том, чтобы по различным характеристикам клиентов спрогнозировать целевую переменную - имел клиент просрочку 90 и более дней или нет (и если имел, то банк не будет выдавать кредит этому клиенту, а иначе будет).

Датасет содержит слежующие признаки:

* `SeriousDlqin2yrs`: клиент имел просрочку 90 и более дней - **целевой признак** (таргет)

* `RevolvingUtilizationOfUnsecuredLines`: общий баланс средств (total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits)

* `age`: возраст заемщика

* `NumberOfTime30-59DaysPastDueNotWorse`: сколько раз за последние 2 года наблюдалась просрочка 30-59 дней

* `DebtRatio`: ежемесячные расходы (платеж по долгам, алиментам, расходы на проживания) деленные на месячный доход

* `MonthlyIncome`: ежемесячный доход

* `NumberOfOpenCreditLinesAndLoans`: количество открытых кредитов (напрмер, автокредит или ипотека) и кредитных карт

* `NumberOfTimes90DaysLate`: сколько раз наблюдалась просрочка (90 и более дней)

* `NumberOfTime60-89DaysPastDueNotWorse`: сколько раз за последние 2 года заемщик задержал платеж на 60-89 дней

* `NumberOfDependents`: количество иждивенцев на попечении (супруги, дети и др)

* `NumberRealEstateLoansOrLines`: количество кредиов (в том числе под залог жилья)

* `RealEstateLoansOrLines`: закодированное количество кредитов (в том числе под залог жилья) - чем больше код буквы, тем больше кредитов

* `GroupAge`: закодированная возрастная группа - чем больше код, тем больше возраст

Решение реализовано на языке python и представлено в виде jupiter-блокнота, который включает в себя разведочный анализ данных, подготовку датасета (обработка пропущенных значений, балансировка категорий и т.д.) и обучение модели.
На выходе получаем файл с весами модели.
