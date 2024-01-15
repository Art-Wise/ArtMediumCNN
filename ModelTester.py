import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import time
import random
from PIL import Image

from ImageProcessing import ImageProcessor
import DatasetParser

# Avoid out of memory errors by seting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class_names = ["Oil", "Watercolor"] #, "Ink"

ImageProcessor = ImageProcessor()
ImageProcessor.target_size = (230, 230)
ImageProcessor.color_space = "RGB"
ImageProcessor.resample_filter = Image.Resampling.LANCZOS

model_name = "art-medium_230x230_RGB-GRAY-EDGE"

print("Loading dataset")
start = time.perf_counter()
(train_images, train_labels), (test_images, test_labels
), (valid_images, valid_labels), (camera_images, camera_labels
), (combine_images, combine_labels) = DatasetParser.ArtMediumDataset(
                                                        ImageProcessor,
                                                        use_cache=True, 
                                                        split=(80, 10, 10),
                                                        random_effect=False,
                                                        edge_channel=True,
                                                        disable_cache=True,
                                                        debug=True)
end = time.perf_counter()
elapsed = end - start
print("Dataset loaded")
print(f"Parsing took: {elapsed:.2f} seconds")

print("Loading model")
start = time.perf_counter()
model = tf.keras.models.load_model(f"Models/{model_name}.keras", safe_mode=False)
end = time.perf_counter()
elapsed = end - start
print("Model loaded")
print(f"Loading took: {elapsed:.2f} seconds")

model.summary()
print(model.get_compile_config())

valid_loss, valid_acc = model.evaluate(valid_images, valid_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)
# camera_loss, camera_acc = model.evaluate(camera_images, camera_labels)
combine_loss, combine_acc = model.evaluate(combine_images, combine_labels)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

graph_images = combine_images
graph_labels = combine_labels

prediction = probability_model.predict(graph_images)

print(f"Test Data:\nModel loss: {test_loss} - Model accuracy: {test_acc}")
print(f"\nValid Data:\nModel loss: {valid_loss} - Model accuracy: {valid_acc}")
# print(f"\nCamera Data:\nModel loss: {camera_loss} - Model accuracy: {camera_acc}")
print(f"\nOverall:\nModel loss: {combine_loss} - Model accuracy: {combine_acc}")


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[:,:,:3].astype(np.uint8), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)
    

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def total_accuracy(oil, watercolor, ink, predictions, labels):
    oil_correct = 0
    watercolor_correct = 0
    ink_correct = 0
    for i in oil:
        true_label = labels[i]
        predicted_label = np.argmax(predictions[i])
        if predicted_label == true_label:
            oil_correct += 1

    for i in watercolor:
        true_label = labels[i]
        predicted_label = np.argmax(predictions[i])
        if predicted_label == true_label:
            watercolor_correct += 1

    for i in ink:
        true_label = labels[i]
        predicted_label = np.argmax(predictions[i])
        if predicted_label == true_label:
            ink_correct += 1

    total_correct = oil_correct + watercolor_correct

    overall_accuracy = (total_correct/(len(oil) + len(watercolor)))*100
    oil_accuracy = (oil_correct/len(oil))*100
    watercolor_accuracy = (watercolor_correct/len(watercolor))*100
    # ink_accuracy = (ink_correct/len(ink))*100
    ink_accuracy = 0
    return oil_accuracy, watercolor_accuracy, ink_accuracy, overall_accuracy


oil = []
watercolor = []
ink = []
for n, label in enumerate(graph_labels):
    match label:
        case 0:
            oil.append(n)
        case 1:
            watercolor.append(n)
        case 2:
            ink.append(n)


images = random.sample(oil, 15) + random.sample(watercolor, 15) #+ random.sample(ink, 10)
num_rows = 6
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

oil_accuracy, watercolor_accuracy, ink_accuracy, overall_accuracy = total_accuracy(oil, watercolor, ink, prediction, graph_labels)
fig = pylab.gcf()
fig.canvas.manager.set_window_title(f"Total Accuracy: {overall_accuracy:2.0f}% - Oil Accuracy: {oil_accuracy:2.0f}% - Watercolor Accuracy: {watercolor_accuracy:2.0f}%")

for i, img in enumerate(images):
    if i < len(graph_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(img, prediction[img], graph_labels, graph_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(img, prediction[img], graph_labels)
plt.tight_layout()
plt.show()

oil = 0
watercolor = 0
ink = 0
for i in train_labels:
    match i:
        case 0:
            oil += 1
        case 1:
            watercolor += 1
        case 2:
            ink += 1

for i in combine_labels:
    match i:
        case 0:
            oil += 1
        case 1:
            watercolor += 1
        case 2:
            ink += 1

print(f"Oil: {oil}\nWatercolor: {watercolor}\nInk: {ink}")
