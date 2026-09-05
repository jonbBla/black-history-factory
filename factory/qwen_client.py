from __future__ import annotations
import json,re
DEFAULT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
def extract_json(text):
    text=(text or "").strip(); text=re.sub(r"^```(?:json)?\s*","",text,flags=re.I); text=re.sub(r"\s*```$","",text)
    try:return json.loads(text)
    except Exception:pass
    for start,c0 in enumerate(text):
        if c0 not in "{[": continue
        stack=[]; ins=False; esc=False
        for i in range(start,len(text)):
            c=text[i]
            if ins:
                if esc: esc=False
                elif c=='\\': esc=True
                elif c=='"': ins=False
                continue
            if c=='"': ins=True
            elif c in "{[": stack.append(c)
            elif c in "}]":
                if not stack or (c=='}' and stack[-1]!='{') or (c==']' and stack[-1]!='['): break
                stack.pop()
                if not stack:
                    try:return json.loads(text[start:i+1])
                    except json.JSONDecodeError: break
    raise ValueError("No valid JSON object or array found in model output.")
class QwenClient:
    def __init__(self,model=None,tokenizer=None,device="cuda"): self.model=model; self.tokenizer=tokenizer; self.device=device; self._quantized=False
    @classmethod
    def load(cls,model_name=DEFAULT_MODEL,device="cuda",load_in_4bit=True):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        import torch
        tok=AutoTokenizer.from_pretrained(model_name); qc=None
        if load_in_4bit and device=="cuda":
            from transformers import BitsAndBytesConfig
            qc=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16 if device=="cuda" else torch.float32,device_map=device,quantization_config=qc)
        if device=="cuda": print(f"Qwen GPU memory after load: {torch.cuda.memory_allocated()/(1024**3):.2f} GB")
        obj=cls(model,tok,device); obj._quantized=bool(qc); return obj
    def generate(self,prompt,max_new_tokens=2048,temperature=0.7):
        messages=[{"role":"user","content":prompt}]; text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); inputs=self.tokenizer([text],return_tensors="pt").to(self.model.device)
        out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,temperature=temperature,do_sample=temperature>0); generated=out[0][inputs["input_ids"].shape[1]:]; return self.tokenizer.decode(generated,skip_special_tokens=True)
    def generate_json(self,prompt,max_new_tokens=2048,retries=1):
        last=None
        for attempt in range(retries+1):
            raw=self.generate(prompt if attempt==0 else prompt+"\n\nReturn ONLY one complete valid JSON object. Start again from scratch; no Markdown or commentary.",max_new_tokens=max_new_tokens,temperature=0.35 if attempt==0 else 0.15)
            try:return extract_json(raw)
            except Exception as e:last=e
        raise ValueError(f"Could not parse valid JSON after {retries+1} attempts: {last}")
