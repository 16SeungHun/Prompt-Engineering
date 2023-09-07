import gc
import torch 
gc.collect()
torch.cuda.empty_cache()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

few_src = [] 
few_tgt = []

def put_datapair(src, tgt):
  few_src.append(src)
  few_tgt.append(tgt)

# k=8
put_datapair("A mountainous area experiencing a wildfire","mountanious area, wildfire.")
put_datapair("A large helicopter dropping water on a wildfire","helocopter, dropping water, wildfire.")
put_datapair("A recovered forest landscape after a wildfire has been extinguished","recovered forest, landscape, wildfire, extinguished")
put_datapair("Volunteers helping with wildfire containment efforts","volunteers, wildfire, containment")
put_datapair("People charging their cars at an electric vehicle charging station","people, electric vehicle, charging station")
put_datapair("Mechanics repairing vehicles at an auto repair shop","mechanics, repairing, vehicles, auto repair shop")
put_datapair("Designers creating vehicle concepts in an automotive design studio","designers, creating vehicle concepts, in automotive design studio")
put_datapair("A parking lot filled with cars of various colors and models","parking lot, cars, various colors and models.")

# ***Reference: [A Recipe for Arbitrary Text Style Transfer with Large Language Models, (Reif et al. (Google Research), ACL 2022)](https://aclanthology.org/2022.acl-short.94/)***
# 탑티어 학회 논문에서 발췌하여 같은 방식의 문장으로 모델을 훈련시킴.
in_context_text = ''

for i in range(len(few_src)):
  in_context_text += f'Here is some text: {few_src[i]}. Here is a rewrite of the text, which is more simple: {few_tgt[i]}\n'

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import pipeline


# GPT-2 (This is the smallest version of GPT-2, with 124M parameters.) = "gpt2"
# GPT-2 (XL) (1.5B parameters.) = "gpt2-xl"
# T5 (base, with 220 million parameters.) = "t5-base"
# T5 (3B parameters.) = "t5-3b"
# T5 (11B parameters.) = "t5-11b" -> 45.2GB (Colab may not cover this size..)
# GPT-J (6B parameters.) = "EleutherAI/gpt-j-6B"

# In-context learning work well at least 1 Billions parameters. -eunchan-

model_zoo = ["gpt2","gpt2-xl","t5-base","t5-3b","t5-11b","EleutherAI/gpt-j-6B"]

model_name = model_zoo[5]

if model_name.find('t5') > -1: #model = t5
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
  nlg_pipeline = pipeline('text2text-generation',model=model, tokenizer=tokenizer)

else: #model = gpt
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  #nlg_pipeline = pipeline(model=model_name, tokenizer=tokenizer)

# model inference

# In[7]:


#test_input_text = 'kjkjkjkj'
test_input_texts = [ "Firefighters battling a raging wildfire",
    "Wild animals fleeing from a wildfire",
    "A forest and ground scorched by a wildfire",
    "Smoke and haze from a wildfire",
    "Village residents evacuating due to a wildfire",
    "A close-up drone image of an ongoing wildfire",
    "A luxury sports car speeding down a highway",
    "A classic car exhibition",
    "Racing cars competing on a race track",
    "Customers filling up their cars at a self-service gas station",
    "A family traveling together in a camper van",
    "Eco-friendly taxi service using electric cars in the city"
]

test_output_length = 32 # token length

for test_input_text in test_input_texts:
    if model_name.find('t5') > -1: # model = T5
        def generate_text(pipe, text, num_return_sequences=5, max_length=512):
            text = f"{text}"
            out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length, num_beams=5, no_repeat_ngram_size=2,)
            return [x['generated_text'] for x in out]

        # target_text = 'kjkjkjkj'
        src_text = in_context_text + f"Here is some text: {test_input_text}\n. Here is a rewrite of the text, which is more simple: "

        print("Input text:", src_text)
        test_output_text = generate_text(nlg_pipeline, src_text, num_return_sequences=1, max_length=test_output_length)

        # you can cook this output anything you want!  
        print(test_output_text)

    else:  # model = GPT
        src_text = in_context_text + f"Here is some text: {test_input_text}\n. Here is a rewrite of the text, which is more simple: "
        tokens = tokenizer.encode_plus(src_text, return_tensors='pt').to(device)
        gen_tokens = model.generate(tokens['input_ids'], attention_mask=tokens['attention_mask'], do_sample=True, temperature=0.8, max_length=len(tokens[0])+test_output_length)
        generated = tokenizer.batch_decode(gen_tokens)[0]

        test_output_text = generated[generated.rfind('more simple:')+12:]
        # print(generated)

        # you can cook this output anything you want!
        print(test_output_text)