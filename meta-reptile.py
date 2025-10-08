from tensorflow import GradientTape, convert_to_tensor, float32
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.losses import  SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.utils import load_img, img_to_array
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score

base_model = load_model('best_model.keras')

# Freezing the layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

n_way = 2
x = base_model.layers[-2].output
output = Dense(n_way, activation='softmax', name='meta_dense')(x)
meta_model = Model(inputs=base_model.input, outputs=output)

loss_fn = SparseCategoricalCrossentropy()
meta_optimizer = Adam(1e-4)


# Sample episodes from the dataset
def sample_episode(dataset, n_way, k_shot, q_query):
    chosen_classes = random.sample(list(dataset.keys()), n_way)
    support_x, support_y, query_x, query_y = [], [], [], []

    for new_label, cls in enumerate(chosen_classes):
        samples = random.sample(dataset[cls], k_shot + q_query)
        s = samples[:k_shot]
        q = samples[k_shot:]
        support_x += [x for x, _ in s]
        support_y += [new_label]*len(s)
        query_x  += [x for x, _ in q]
        query_y  += [new_label]*len(q)

    return np.array(support_x), np.array(support_y), np.array(query_x), np.array(query_y)


# Reptile InnerLoop Training
def inner_train(model, support_x, support_y, inner_steps=5, inner_lr=1e-3):
    inner_model = clone_model(model)
    inner_model.set_weights(model.get_weights())
    opt = Adam(inner_lr)

    for _ in range(inner_steps):
        with GradientTape() as tape:
            preds = inner_model(support_x, training=True)
            loss = loss_fn(support_y, preds)
        grads = tape.gradient(loss, inner_model.trainable_variables)
        opt.apply_gradients(zip(grads, inner_model.trainable_variables))
    return inner_model

# Reptile Meta-Update
def reptile_update(model, inner_model, meta_step_size=0.1):
    new_weights = []
    for w_meta, w_inner in zip(model.get_weights(), inner_model.get_weights()):
        new_w = w_meta + meta_step_size * (w_inner - w_meta)
        new_weights.append(new_w)
    model.set_weights(new_weights)


# Meta Training Loop
def meta_train(meta_model, dataset, meta_iters=1000, n_way=2, k_shot=5, q_query=10, inner_steps=5, inner_lr=1e-3, meta_step_size=0.1):
    for it in range(meta_iters):
        # Sample episode
        support_x, support_y, query_x, query_y = sample_episode(dataset, n_way, k_shot, q_query)

        # Convert to tensors
        support_x = convert_to_tensor(support_x, dtype=float32)
        query_x   = convert_to_tensor(query_x, dtype=float32)

        # Inner adaptation
        inner_model = inner_train(meta_model, support_x, support_y, inner_steps, inner_lr)

        # Meta update
        reptile_update(meta_model, inner_model, meta_step_size)

        # Evaluate on query set occasionally
        if (it + 1) % 50 == 0:
            preds = meta_model(query_x, training=False).numpy()
            pred_labels = np.argmax(preds, axis=1)
            acc = np.mean(pred_labels == query_y)
            print(f"Iter {it + 1}: Episodic Query Accuracy = {acc * 100:.2f}%")

    meta_model.save("meta_reptile.keras")
    print("Meta-trained model saved as meta_reptile.keras")


# Few-Shot Adaption + Evaluation
def adapt_and_eval(meta_model, support_x, support_y, query_x, query_y, adapt_steps=20, adapt_lr=1e-4):
    adapted_model = clone_model(meta_model)
    adapted_model.set_weights(meta_model.get_weights())
    opt = Adam(adapt_lr)

    # Adapt on support set
    for _ in range(adapt_steps):
        with GradientTape() as tape:
            preds = adapted_model(support_x, training=True)
            loss = loss_fn(support_y, preds)
        grads = tape.gradient(loss, adapted_model.trainable_variables)
        opt.apply_gradients(zip(grads, adapted_model.trainable_variables))

    # Evaluate on query
    preds = adapted_model(query_x, training=False).numpy()
    pred_labels = np.argmax(preds, axis=1)
    acc = np.mean(pred_labels == query_y)

    with open('results/meta_reptile_report.txt', 'a') as f:
        f.write("Accuracy of the Meta-Reptile Model = {acc * 100:.2f}%\n")


train_dir = "data/pediatric/train"
test_dir = "data/pediatric/test"


def build_meta_dataset(base_dir, target_size=(224,224)):
    dataset = {}
    class_names = sorted(os.listdir(base_dir))  # each subfolder = class
    for class_idx, cls in enumerate(class_names):
        cls_path = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        dataset[class_idx] = []
        for fname in os.listdir(cls_path):
            img_path = os.path.join(cls_path, fname)
            try:
                img = load_img(img_path, target_size=target_size)
                arr = img_to_array(img) / 255.0   # rescale like your datagen
                dataset[class_idx].append((arr, class_idx))
            except:
                pass
    return dataset


train_dataset = build_meta_dataset(train_dir)
test_dataset  = build_meta_dataset(test_dir)


meta_train(meta_model, train_dataset, meta_iters=500, n_way=2, k_shot=5, q_query=10)


support_x, support_y, query_x, query_y = sample_episode(test_dataset, 2, 5, 15)
support_x = convert_to_tensor(support_x, float32)
query_x   = convert_to_tensor(query_x, float32)


adapt_and_eval(meta_model, support_x, support_y, query_x, query_y)