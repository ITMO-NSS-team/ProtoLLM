# PROTO_LLM-SDK

## Введение

PROTO_LLM-SDK - это набор инструментов для разработки и интеграции в рамках платформы PROTO_LLM

Для интеграции внешний модуль должен соответствовать следующим требованиям:

1) Отсутствие операций чтения/записи в файловую систему
2) В проекте используется poetry для управления зависимостями ([туториал](https://habr.com/ru/articles/593529/),
   [еще туториал](https://habr.com/ru/articles/740376/))
3) Содержательная часть проекта должна находится в директории с названием соответствующим
   названию модуля объявленному в pyproject.toml. Все дополнительные файлы, эксперименты, изображения, тесты должны
   быть вынесены в отдельные директории.
   Например:
    ```
    stairs-sdk/ # репозиторий / корень проекта
    ├── stairs_skd/
    │   ...
    ├── tests/
    │   ...
    ├── experements/
    │   ...
    ├── README.md
    ├── pyproject.toml
    ├── poetry.lock
    ...
   ```
4) Все импорты внутри репозитория настраиваются от директории в которой содержится модуль.
   Например:
    ```python
    from stairs_sdk.sdk.job_context.job import Job
    import stairs_sdk.sdk.job_context.job
   ```
   а не
    ```python
    from sdk.job_context.job import Job
    ```
4) Все зависимости должны быть указаны в файле pyproject.toml
5) Все зависимости не связанные с инференсом должны находится в dev разделе pyproject.toml (например, pytest, jupyter, tqdm и т.д.)
6) Установлен репозиторий stairs-sdk в модуль

## Установка

Для установки SDK нужно инициализировать poetry в проекте (если он не был инициализирован ранее)

```bash
poetry init
```

После чего добавить все зависимости в pyproject.toml

```bash
poetry add $(cat requirements.txt)
```

После выполнить команды

```bash
poetry lock --no-update
poetry install
```

Если возникнут ошибки, то поменять на команды

```bash
poetry lock --no-update
poetry config experimental.system-git-client true
poetry install
```

## Интеграция

Общий пайплайн процесса интеграции внешнего модуля в PROTO_LLM выглядит следующим образом:

1) Создание класса наследуемого от `Job` и реализация метода `run`
2) Согласование с командой PROTO_LLM параметров входа с примерами и аннотациями
3) Согласование с командой PROTO_LLM выходного словаря сохраняемого в Redis с примерами и аннотациями
4) Если внутри модуля требуется вызов других сервисов PROTO_LLM, то необходимо такое взаимодействие согласовывать с
   командой PROTO_LLM
   (на текущем этапе вся логика взаимодействия в разных сервисов будет на стороне PROTO_LLM)

Для каждой отдельной функциональности модуля вносимой наружу
(например для валидации отдельными будут быстрая валидация и полная валидация)
создаем отдельный класс наследуемый от `Job` и реализуем в нем метод `run`. 
Каждая такая функциональность единоразово принимает на вход данные,
использует в процессе выполнения служебные сервисы (LLM, LLMLongChain, Embedder, VectorDB, etc)
и сохраняет в Redis результаты выполнения с помощью инструментов реализованных в SDK.
Модели для сохранения в Redis должны быть описаны через pydantic модели из базовых типов сделанные в ООП стиле. 
То есть не допускается использование вложенной типизации в pydantic моделях типа 
`pd.DataFrame` или `list[dict[str, Union[str, float]]]`.

Дополнительные условия:
1) Нужно README.md с описанием модуля используя шаблон [README](sdk/sdk/job_context/README.md)
2) Нейминг Job: `ModuleNameJob` если Job в модуле одна, `ModuleNameFeature1Job`, `ModuleNameFeature2Job` и т.д. если Job несколько
3) Нейминг моделей результатов: `ModuleNameResult`, `ModuleNameFeature1Result`, `ModuleNameFeature2Result` и т.д. 
Вложенные модели должны сохранять логику нейминга родительской модели, например `ModuleNameItem`, `ModuleNameAttribute1` и т.д.
4) В корне модуля (не путать с корнем репозитория) создать файл jobs.py, 
в котором будут импортированы все Job классы и все модели результатов (Job и модели результатов могут быть реализованы где угодно внутри модуля).


## Примеры

1) Простой пример:

```python
from stairs_sdk.sdk.job_context.job import Job
from stairs_sdk.sdk.job_context.job_context import JobContext
from stairs_sdk.sdk.models.job_context_models import ( PromptModel, PromptMeta, TextEmbedderRequest, 
                                                       ChatCompletionModel, ChatCompletionUnit)

from pydantic import BaseModel

from my_module import my_function1, my_function2


class ExampleResult(BaseModel):
    # Определение модели результата
    result: str


class ExampleJob(Job):
    def run(self, job_id: str, ctx: JobContext, **kwargs):
        # Извлечение параметров из kwargs, все ваши параметры должны быть переданы в kwargs
        param1 = kwargs.get("param1")
        param2 = kwargs.get("param2")

        # Выполнение ваших функций
        result1 = my_function1(param1)

        # формирование коммуникационной модели для векторизации текста и выполнение запроса векторизации
        vectorized = ctx.text_embedder.inference(TextEmbedderRequest(job_id=job_id, inputs=result1))

        # Выполнение ваших функций
        result2 = my_function2(vectorized, param2)

        # Выполнение запроса к LLM
        request_model = PromptModel(job_id=job_id, content=result2, meta=PromptMeta(temperature=0.2))
        result_final: str = ctx.llm_api.inference(request_model).content
        # пример запроса ко VSEGPT к gpt4o
        result_final_outer: str = ctx.outer_llm_api.inference(request_model).content 
        # пример запроса ко VSEGPT к llama (к произвольной модели)
        request_model = PromptModel(job_id=job_id, content=result2, meta=PromptMeta(temperature=0.2, model="meta-llama/llama-3.1-70b-instruct"))
        result_final_outer_llama: str = ctx.outer_llm_api.inference(request_model).content 
        
        # Выполнение запроса к LLM с использованием chat_completion структуры
        request_model = ChatCompletionModel(
            job_id=job_id, 
            messages=[
                ChatCompletionUnit(
                    role='system',
                    content='%system_prompt%'
                ),
                ChatCompletionUnit(
                    role='user',
                    content=result2
                )
            ], 
            meta=PromptMeta(temperature=0.2))
        result_completion: str = ctx.llm_api.chat_completion(request_model).content
        # пример запроса ко VSEGPT
        result_completion_outer: str = ctx.outer_llm_api.chat_completion(request_model).content
        

        # Сохранение результата в Redis
        ctx.result_storage.save_dict(job_id, ExampleResult(result=result_final).dict())
        ctx.result_storage.save_dict(job_id, ExampleResult(result=result_completion).dict())

```

2) Пример коммуникационной модели со вложениями на примере pd.DataFrame

```python
from pydantic import BaseModel


class ExamplePandasItem(BaseModel):
    # Определение модели строки
    name: str
    value: float


class ExamplePandasResult(BaseModel):
    # Определение модели DataFrame
    units: list[ExamplePandasItem]

    # Некорректное использование
    # units: pd.DataFrame
    # or
    # units: list[dict[str, Union[str, float]]] 
 ```

Для локального тестирования необходимо:

1. окружение
2. сгенерить job_id через uuid.uuid4()
3. сгенерить JobContext через construct_job_context(...)
4. Запустить метод run с job_id и JobContext
5. Проверить результаты выполнения

Шаг 1. Окружение

```bash
PROTO_LLM_LLM_API_HOST = 10.32.15.21
PROTO_LLM_LLM_API_PORT = 6672

PROTO_LLM_TEXT_EMB_HOST = 10.32.15.21
PROTO_LLM_TEXT_EMB_PORT = 9942

PROTO_LLM_REDIS_HOST = 10.32.15.21
PROTO_LLM_REDIS_PORT = 6379

PROTO_LLM_RABBIT_HOST = 10.32.15.21
PROTO_LLM_RABBIT_PORT = 5672
```

Note: Если вы работаете в PyCharm, то можно добавить эти переменные в
Run/Debug Configuration -> Environment variables ->
`PROTO_LLM_LLM_API_HOST=10.32.15.21;PROTO_LLM_LLM_API_PORT=6672;PROTO_LLM_TEXT_EMB_HOST=10.32.15.21;PROTO_LLM_TEXT_EMB_PORT=9942;PROTO_LLM_REDIS_HOST=10.32.15.21;PROTO_LLM_REDIS_PORT=6379;PROTO_LLM_RABBIT_HOST=10.32.15.21;PROTO_LLM_RABBIT_PORT=5672`

Шаги 2-5. Пример запуска

```python
import uuid
from stairs_sdk.sdk.job_context.utility import construct_job_context
from stairs_sdk.utils.reddis import get_reddis_wrapper, load_result

# step 2
job_id = str(uuid.uuid4())

# step 3
job_name = "fast_validation"  # or "full_validation" or "other_job_name" which describe type of job
ctx = construct_job_context(job_name)

# step 4
ExampleJob().run(job_id, ctx, param1="value1", param2="value2")

# step 5
rd = get_reddis_wrapper()
result = load_result(rd, job_id, job_name)
```