from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

prompt1 = "Hey, how are you doing today?"
prompt2 = "Good morning."
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."

input_ids = tokenizer([description, description], return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer([prompt1, prompt2], padding=True, truncation=True,  return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
print(generation.shape)
audio_arr = generation.cpu().numpy().squeeze()
print(audio_arr.shape)
for i in range(2):
    sf.write(f"{i}_parler_tts_out.wav", audio_arr[i].squeeze(), model.config.sampling_rate)