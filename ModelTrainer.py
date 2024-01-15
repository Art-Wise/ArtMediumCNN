import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import time
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

# Only name of the model, no path or ".keras" e.g.(art-medium_230x230_RGB-GRAY-EDGE)
model_name = "art-medium_230x230_RGB-GRAY-EDGE"

# Load Dataset
print("Loading dataset")
start = time.perf_counter()
(train_images, train_labels), (test_images, test_labels
), (valid_images, valid_labels), (camera_images, camera_labels
), (combine_images, combine_labels) = DatasetParser.ArtMediumDataset(
                                                        ImageProcessor,
                                                        use_cache=True, 
                                                        split=(80, 10, 10),
                                                        random_effect=True,
                                                        edge_channel=True,
                                                        disable_cache=True,
                                                        debug=True)
end = time.perf_counter()
elapsed = end - start
print("Dataset loaded")
print(f"Parsing took: {elapsed:.2f} seconds")

better = False
for i in range(5): # Number of training iterations
    print(f"Training number {i+1}")

    # Load Model
    print("Loading model")
    start = time.perf_counter()
    model = tf.keras.models.load_model(f"Models/{model_name}.keras", safe_mode=False) # Safe mode diabled to enable the use of Lambda
    end = time.perf_counter()
    elapsed = end - start
    print("Model loaded")
    print(f"Loading took: {elapsed:.2f} seconds")

    model.summary()
    print(model.get_compile_config())

    logdir = "Models/Logs/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    checkpoint_filepath = f"Models/Checkpoints/{model_name}_weights_T{i+1}.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        save_freq="epoch"
    )

    model.fit(
        train_images,
        train_labels,
        epochs=50,
        batch_size=110,
        callbacks=[model_checkpoint, tensorboard_callback],
        validation_data = (valid_images, valid_labels),
        shuffle=True,
    )

    model = tf.keras.models.load_model(checkpoint_filepath, safe_mode=False)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    valid_loss, valid_acc = model.evaluate(valid_images, valid_labels)
    # camera_loss, camera_acc = model.evaluate(camera_images, camera_labels)
    combine_loss, combine_acc = model.evaluate(combine_images, combine_labels)
    print(f"Best weights:\nval_loss: {combine_loss} - val_accuracy: {combine_acc}")

    old_model = tf.keras.models.load_model(f"Models/{model_name}.keras", safe_mode=False)
    old_test_loss, old_test_acc = old_model.evaluate(test_images, test_labels)
    old_valid_loss, old_valid_acc = old_model.evaluate(valid_images, valid_labels)
    # old_camera_loss, old_camera_acc = old_model.evaluate(camera_images, camera_labels)
    camera_acc = 0.0
    old_combine_loss, old_combine_acc = old_model.evaluate(combine_images, combine_labels)

    print(f"Old Model:\nTest Loss: {old_test_loss} - Test Accuracy: {old_test_acc}\nNew Model:\nTest Loss: {test_loss} - Test Accuracy: {test_acc}")
    print(f"\nOld Model:\nValid Loss: {old_valid_loss} - Valid Accuracy: {old_valid_acc}\nNew Model:\nValid Loss: {valid_loss} - Valid Accuracy: {valid_acc}")
    # print(f"\nOld Model:\nCamera Loss: {old_camera_loss} - Camera Accuracy: {old_camera_acc}\nNew Model:\nCamera Loss: {camera_loss} - Camera Accuracy: {camera_acc}")
    old_camera_acc = 0.0
    print(f"\nOld Model:\nCombine Loss: {old_combine_loss} - Combine Accuracy: {old_combine_acc}\nNew Model:\nCombine Loss: {combine_loss} - Combine Accuracy: {combine_acc}")

    # Only save new weights if the model improved
    # If the overall accuracy is greater and the test and valid accuracies haven't decreased by more than 2%
    # Might not be the best way to do this but i've had good results while using it (Without this a model may get worse after training)
    if (combine_acc > old_combine_acc) and (old_test_acc - test_acc <= 0.02 and old_valid_acc - valid_acc <= 0.02):
        model.save(f"Models/{model_name}.keras")
        print("New model weights saved.")
        better = True
    else:
        print("Old model is better.")


if better:
    print("Model was improved!")
