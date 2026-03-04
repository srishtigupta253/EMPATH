This dataset helps models attain affective empathy

On acceptance, we wish to deploy the final model code here, where the model is already available on HuggingFace with a CC-by-4.0 License.
```
# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline(``text2text-generation'', model="author/EMPATH_medium")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(``author/EMPATH_medium'')
model = AutoModelForSeq2SeqLM.from_pretrained(``author/EMPATH_medium'')
```
