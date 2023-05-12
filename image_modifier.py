from transformers import OpenAiAgent
from creds import openai_key, hf_key
from playsound import playsound
import soundfile as sf
from huggingface_hub import login
from transformers import HfAgent
from PIL import Image

# Either use openai agent, or
agent = OpenAiAgent(model="text-davinci-003", api_key=openai_key)

image = Image.open('path/to/image.jpg')
image.show()

updated_image = agent.run("Transform the image in `image` to add fire to it.", image=image)
updated_image.show()
