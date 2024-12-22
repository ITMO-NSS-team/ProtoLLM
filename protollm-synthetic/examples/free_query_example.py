import os
from samplefactory.synthetic_pipelines.chains import QuizChain
from samplefactory.utils import Dataset, VLLMChatOpenAI
import json

# создаем игрушечный датасет из вырезок случайных статей википедии
texts = [
"Сегодня хорошая погода в Москве",
"Завтра в 11 будет важный созвон с заказчиком из Италии",
"Я записан в бассейн 30.12.2024 в 12 и 31.12.2024 в 13. Удивляюсь, как мне это удалось",
"Через неделю будет вечеринка в клубе 'Золотой' в 23:00",
"Сегодня в 10:00 я занялся спортом",
"19.01.2025 я записан к стоматологу"
]

solutions = [
    "[{'date': '22.12.2024', 'event': 'хорошая погода в Москве'}]",
    "[{'date': '23.12.2024', 'time': '11:00', 'event': 'созвон с заказчиком из Италии'}]",
    "[{'date': '30.12.2024', 'time': '12:00', 'event': 'запись в бассейн'}, {'date': '31.12.2024', 'time': '13:00', 'event': 'запись в бассейн'}]",
    None,
    None,
    None,
    ]

data_dict = {'content': texts, 'solution': solutions}

with open('tmp_data/sample_data_free_instruction.json', 'w', encoding='utf-8') as file:
    json.dump(data_dict, file, ensure_ascii=False)

dataset = Dataset(data_col='content', labels_col='solution', path='tmp_data/sample_data_free_instruction.json')

# TODO write everything
