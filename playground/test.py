from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.preprocessors import TokenClassificationTransformersPreprocessor

model_id = 'iic/nlp_structbert_word-segmentation_chinese-base'
model = Model.from_pretrained(model_id)
tokenizer = TokenClassificationTransformersPreprocessor(model.model_dir)
pipeline_ins = pipeline(task=Tasks.word_segmentation, model=model, preprocessor=tokenizer)

inputs = [
    '人要是行，干一行行一行，一行行行行行，行行行干哪行都行。要是不行，干哪行一行不行一行，一行不行行行不行，行行不行干哪行都不行。要想行行行，首先一行行。成为行业内的内行，行行成内行，行行行。你说我说的行不行。',
    '结婚的和尚未结婚的。',
    '南京市长江枢纽欢迎您。',
    '做爱做的事，交配交的人。',
]
result = pipeline_ins(inputs, batch_size=64)
print(result)
