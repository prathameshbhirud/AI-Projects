from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
model = PeftModel.from_pretrained(base_model, "./resume_lora")

model = model.merge_and_unload()

model.save_pretrained("./resume_final")