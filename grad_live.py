import gradio as gr
import numpy as np
import torch
import model
import helper
import MyDataSet
import model_probit
import gradio as gr

lattent_2class = model_probit.latent_space()
lattent_2class = helper.import_model_name(model_x=lattent_2class, activate_eval=True)
model_vae = model.VaeFinal_only_one_hidden_copy()
model_vae = helper.import_model_name_weights_copy(model_x=model_vae, activate_eval=True)
# MyModel5_retry_2classes_faster_1
classifier = model.MyModel5_retry_2classes_faster_1()
classifier = helper.import_model_name(model_x=classifier, activate_eval=True)

def predict_input(img) :
    numpy_to_tensor = torch.from_numpy(img).view(1,-1)
    print(numpy_to_tensor.shape)
    return classifier(img)
    # return classifier(img).argmax(dim=0)
with gr.Blocks() as demo:
    gr.Markdown("# DEMO")
    steps = gr.Textbox(placeholder="How many steps do you want to go towards the other class?")
    draw = gr.Sketchpad(type='numpy',shape=(28,28),image_mode = 'L')
    # out = gr.Textbox(predict_input(draw))

    draw.change(fn=predict_input,
               inputs=draw,
               outputs=gr.Textbox())

if __name__ == "__main__":
    demo.launch()