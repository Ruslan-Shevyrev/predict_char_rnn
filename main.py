import nn
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

phrase = 'Россия'

(tokenizer, model) = nn.get_model(address='https://ru.wikipedia.org/wiki/%D0%A0%D0%BE%D1%81%D1%81%D0%B8%D1%8F')
res = nn.build_phrase(phrase, tokenizer, model)
print(res)
