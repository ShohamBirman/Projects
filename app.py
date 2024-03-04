import gradio as gr
from fastai.vision.all import *


def classify_img(img):
    pred, idx, probs = model.predict(img)
    return dict(zip(model.dls.vocab, probs))

image = gr.Image()
label = gr.Label()
examples = ['examples/grizzly.jpg', 'examples/black.jpg', 'examples/teddy.jpg']

model = load_learner('export.pkl')

intf = gr.Interface(fn=lambda image: "grizzly bear", inputs=image, outputs=label, examples=examples)
iface = gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
iface.launch()