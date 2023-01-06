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


def ret_w_w0():
    iterweights = iter(lattent_2class.parameters())
    w = next(iterweights).data[0]
    w0 = next(iterweights).data[0]
    return w, w0


with torch.no_grad():
    dataset_49 = MyDataSet.MyDataSets_Subset_4_9(batch_size_train=-1)
    img_49_batch, label_49_batch = next(iter(dataset_49.train_loader_subset_changed_labels))
    rec49, mu49, sigma49 = model_vae(img_49_batch.clone())
    z = mu49
    w, w0 = ret_w_w0()


def return_counter_with_steps(z, steps=20, w=w, w0=w0):
    STEPS = steps
    alpha_i = - (torch.t(z) * w + w0) / torch.t(w) * w
    z_counter = z + STEPS * alpha_i * w
    with torch.no_grad():
        z_counter_recons_img = model_vae.decode(z_counter)
    return z_counter_recons_img.view(28, 28)


def predict(index, steps):
    print(index, steps)
    index = int(index)
    steps = int(steps)
    z_counter_recons_img = return_counter_with_steps(z=z[index], steps=steps)
    counter_image = z_counter_recons_img
    counter_image = counter_image.clamp(0, 1)
    # print(f'{counter_image = }')
    return return_image_in_format(counter_image)


def return_image_in_format(counter_image):
    counter_image = np.asarray(counter_image)
    return counter_image


def show_counterfactual(index, steps):
    outputs = gr.Image(image_mode='L', shape=(28, 28), value=np.asarray(predict(index, steps))),
    return outputs


demo = gr.Interface(
    live=True,
    fn=predict,
    inputs=[gr.Number(label="index of z"), gr.Number(label="steps towards border")],
    outputs='image',

    title="Counterfactuals",
    description="pick an image, show reconstruction, steps for counterfactuals!",
)
demo.launch()
