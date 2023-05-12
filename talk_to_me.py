from transformers import OpenAiAgent
from creds import openai_key, hf_key
from playsound import playsound
import soundfile as sf
from huggingface_hub import login
from transformers import HfAgent

# Either use openai agent, or
agent = OpenAiAgent(model="text-davinci-003", api_key=openai_key)

# Use HF agent (kinda bad atm)
#login(hf_key)
# Starcoder
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
#agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")

audio = agent.run("Tell me who runs the world")

sf.write("audio/speech_converted.wav", audio.numpy(), samplerate=16000)
playsound("audio/speech_converted.wav")
