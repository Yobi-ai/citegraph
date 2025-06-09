import gradio as gr
import requests

BASE_URL = "http://127.0.0.1:8001"
HEADERS = {"Content-Type": "application/json"}


def load_documents(file_path):
    print(file_path)
    url = f"{BASE_URL}/api/predict"
    if file_path is not None:
        with open(file_path.name, "rb") as f:
            file = {"file": f}
            response = requests.post(url, files=file)
        f.close()

    output = response.json()
    print(output)
    return output["predicted_label"]


def createUI():
    with gr.Blocks() as demo:
        with gr.Row():
            class_label = gr.Markdown("")
        with gr.Row():
            with gr.Column(scale=1):
                # pdf_display = gr.Image(
                #     label="Uploaded PDF Page", interactive=False, height=680
                # )
                upload_btn = gr.File(label="Upload a PDF", file_types=[".pdf"])
                upload_btn.upload(
                    fn=load_documents, inputs=[upload_btn], outputs=[class_label]
                )
    return demo


if __name__ == "__main__":
    demo = createUI()
    demo.launch(pwa=True, share=False)
