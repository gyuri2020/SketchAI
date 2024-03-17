import gradio as gr
import time
from PIL import Image


def get_sketch(im):
    im_pil = im['composite']

    return im_pil


with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column():
            im = gr.ImageEditor(
                type="pil",
                crop_size="1:1",
            )
        with gr.Column():
            sketch = gr.Image(type="pil")


    im.change(fn=get_sketch, outputs=sketch, inputs=im)

if __name__ == "__main__":
    demo.launch()

