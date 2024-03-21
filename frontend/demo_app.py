import gradio as gr
from backend.utils.utils import get_sketch_for_image, get_sketch_for_text, show_sketch_info

with gr.Blocks() as demo:
    gr.HTML("<h1> SketchAI </h1>")
    gr.HTML("<h2> AI가 1만여명 중 당신과 가장 이미지가 비슷한 사람을 찾고, 이미지를 설명해드릴게요! </h2>")
    gr.HTML("<h3> 아니면 원하는 얼굴 분위기, 스타일을 텍스트로 검색해보세요. AI가 해당하는 얼굴 이미지를 제공합니다. </h3>")
    
    with gr.Row():
        with gr.Column(scale=4):
            im = gr.ImageEditor(
                label="당신의 이미지를 업로드 해주세요",
                type="pil",
                crop_size="1:1",
            )
            text = gr.Textbox(label="찾고 싶은 얼굴 이미지를 말하고 Enter를 눌러주세요", info="원하는 분위기, 헤어 스타일 모든 검색하세요. 이미지로 찾아드릴게요.",placeholder="Enter text here...")
        with gr.Column(scale=2):
            result_img = gr.Image(label="1만여명 중 당신과 가장 비슷한 이미지의 사람은?",type="pil")       
        with gr.Column(scale=2):
            result_sketch = gr.Image(label="1만여명 중 당신과 가장 비슷한 이미지의 사람은?",type="pil")
    
    with gr.Row():
        choices = ['face', 'hairstyle', 'eyebrows', 'eyes', 'nose', 'mouth', 'neck', 'wrinkle', 'feature', 'impression']
        category_drop_down = gr.Dropdown(label="설명을 원하는 카테고리를 골라주세요", info="어떤게 궁금하세요?", type="value", choices=choices)

        info_box = gr.JSON(label="선택한 카테고리에 대한 설명입니다.")

    im.change(fn=get_sketch_for_image, outputs=[result_img,result_sketch ], inputs=[im])
    text.submit(fn=get_sketch_for_text, outputs=[result_img,result_sketch ], inputs=[text])
    category_drop_down.change(fn=show_sketch_info, outputs=[info_box], inputs=[category_drop_down])

if __name__ == "__main__":
    demo.launch()

