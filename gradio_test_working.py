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

def predict_label(img):
    return classifier(torch.tensor(img).view(1,1,28,28))


def return_counter_with_steps(z, steps=20, w=w, w0=w0):
    STEPS = steps
    alpha_i = - (torch.t(z) * w + w0) / torch.t(w) * w
    z_counter = z + STEPS * alpha_i * w
    with torch.no_grad():
        z_counter_recons_img = model_vae.decode(z_counter)
    # return z_counter_recons_img.view(-1,28, 28)
    print(f'{z_counter_recons_img = }')
    return z_counter_recons_img


def predict(index, steps, show_original):
    if show_original:
        return show_original(index)
    print(index, steps)
    index = int(index)
    steps = int(steps)
    z_counter_recons_img = return_counter_with_steps(z=z[index], steps=steps)
    counter_image = z_counter_recons_img
    counter_image = counter_image.clamp(0, 1)
    # print(f'{counter_image = }')
    return return_image_in_format(counter_image)


def show_original(index):
    outputs = gr.Image(image_mode='L', shape=(1,28*28), value=img_49_batch[index]),
    return outputs


def return_image_in_format(counter_image):
    counter_image = np.asarray(counter_image)
    print(counter_image.shape)

    return counter_image


def show_counterfactual(index, steps):
    outputs = gr.Image(image_mode='L', shape=(28, 28), value=np.asarray(predict(index, steps))),
    return outputs



def predict_drawing(steps, z):
    steps = int(steps)
    z_counter_recons_img = return_counter_with_steps(z=z, steps=steps)
    counter_image = z_counter_recons_img
    counter_image = counter_image.clamp(0, 1)
    # print(f'{counter_image = }')
    return counter_image
    # return return_image_in_format(counter_image)



def draw_and_show_counterfactual_and_label(steps, image: np):
    steps = int(steps)
    image_float = image / 255.
    tensor_img = torch.tensor(image_float, dtype=torch.float)
    print(f'{tensor_img.shape = }')
    print(f'{tensor_img.dtype = }')
    print(f'{tensor_img = }')
    img_rec, mu, sigma = model_vae(tensor_img.view(2,1,28*28))
    z = mu
    print(f'{z  = }')
    counter = predict_drawing(steps, z)
    # print(f'{counter = }')
    # print(f'{counter.shape = }')
    # counter = counter.view(-1,1,28,28)
    # print(f'{counter = }')
    # print(f'{counter.shape = }')
    # counter = counter[0]
    # print(f'{counter = }')
    # print(f'{counter.shape = }')
    # print(f'{counter.dtype = }')
    # counter = np.asarray(counter)
    # print('-------')
    # print(f'{counter = }')
    # print(f'{counter.shape = }')
    # print(f'{counter.dtype = }')
    counter = counter
    print(counter.shape)
    outputs = gr.Image(image_mode='L', shape=(28, 28), value=counter),
    return outputs

# image = gr.Image(image_mode='L', shape=(28, 28), value=np.zeros((28, 28)))

demo = gr.Interface(
    live=True,
    fn=predict,
    inputs=[gr.Number(label="index of z"), gr.Number(label="steps towards border")],
    outputs='image',

    title="Counterfactuals",
    description="pick an image, show reconstruction, steps for counterfactuals!",
)
demo2 = gr.Interface(
    live=True,
    fn=draw_and_show_counterfactual_and_label,
    inputs=[gr.Number(label="steps towards border"),
            gr.Sketchpad(shape=(28, 28), image_mode='L', type='numpy')],
    outputs='image',

    title="Counterfactuals",
    description="pick an image, show reconstruction, steps for counterfactuals!",
)

# tensor_image = torch.tensor(image)
# image = gr.Image(image_mode='L', shape=(28, 28), value=np.zeros((28, 28)))
demo_predict_written = gr.Interface(
    # live=True,
    fn=predict_label,
    inputs=[gr.Sketchpad(shape=(28, 28), image_mode='L', type='numpy')]
,
    outputs='number',

    title="Counterfactuals",
    description="pick an image, show reconstruction, steps for counterfactuals!",
)

demo_predict_written.launch()
