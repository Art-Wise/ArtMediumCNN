import numpy as np
from PIL import Image
import time
import pickle
import os
import cv2

from ImageProcessing import ImageProcessor


def EdgeDetection(cv_img):
    cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv_blur = cv2.GaussianBlur(cv_gray, (3,3), 0)
    cv_edges = cv2.Sobel(src=cv_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    return cv_edges


def ArtMediumDataset(img_pro, train_hold=None, use_cache=True, split=(80, 10, 10), random_effect=False, gray_channel=True, edge_channel=False, build=True, disable_cache=True, debug=False):
    """
    Returns datasets as tuples.\n
    (train_images, train_labels), (test_images, test_labels), (valid_images, valid_labels), (camera_images, camera_labels), (combine_images, combine_labels)\n
    img_pro = List of 2 ImageProcessors e.x. [ImageProcessor()]\n
    train_hold = int between 1 and 99\n
    use_cache = True or False\n
    split = tuple of percentages as ints (train, test, valid)
    disable_cache = True or False \n
    train_hold: The percentage of data you want to leave out of the training data.\n
    NOTE: train_hold is currently not used.\n
    gray_channel: Add a gray channel to the image.\n
    edge_channel: Add a edge channel to the image.\n
    use_cache: Will attempt to use cached datasets if they exist. If the requested dataset does not exist it will create it.(If data has been updated set this value to False in order to force an update.)\n
    disable_cache: If True will not save cache files. Cache files are very large. (9GB+ with edge_channel and random_effect enabled with 1,200 images.)\n
    NOTE: Each setting will have its own cache.
    """

    class_names = ["oil", "watercolor", "ink"]

    if split[0] + split[1] + split[2] == 100.0 and len(split) == 3:
        pass
    else:
        if len(split) != 3:
            print("Split tuple is not a length of 3, setting to default (80%, 10%, 10%)")
        else:
            print("Split tuple is not equal to 100, setting to default (80%, 10%, 10%)")
        split = (80, 10, 10)

    filepath = (f"Datasets/ArtMediumDataset-"
                f"{img_pro.target_size[0]}x{img_pro.target_size[1]}_{img_pro.color_space}"
                f"{'&Gray' if gray_channel else ''}"
                f"{'&Edge' if edge_channel else ''}"
                f"_{img_pro.resample_filter.name}"
                f"_{split[0]},{split[1]},{split[2]}"
                f"{'_random' if random_effect else ''}.pkl")

    if debug:
        print(filepath)
                            
    if use_cache and os.path.isfile(filepath):
        ArtMediumData = pickle.load(open(filepath, "rb"))

    if use_cache and "ArtMediumData" in locals():
        # Target = (list of np.arrays, list of labels)
        # Load data
        train_images, train_labels = ArtMediumData["train"]
        (train_images, train_labels) = ([np.array(img) for img in train_images], train_labels)

        (test_images, test_labels) = ArtMediumData["test"]
        (test_images, test_labels) = ([np.array(img) for img in test_images], test_labels)

        (valid_images, valid_labels) = ArtMediumData["valid"]
        (valid_images, valid_labels) = ([np.array(img) for img in valid_images], valid_labels)

        (camera_images, camera_labels) = ArtMediumData["camera"]
        (camera_images, camera_labels) = ([np.array(img) for img in camera_images], camera_labels)
        using_cache = True

    else:
        using_cache = False
        class_dict = {}
        folders = {"oil": ["Data/Google/Oil Paintings"], "watercolor": ["Data/Google/Watercolor Paintings"]}
        for material in [*folders]:
            for folder in folders[material]:
                for file in os.listdir(folder):
                    if material in class_dict:
                        class_dict[material].append(f"{folder}/{file}")
                    else:
                        class_dict[material] = [f"{folder}/{file}"]
            

        split = (split[0] * .01, split[1] * .01, split[2] * .01)
        train_oil, test_oil, valid_oil = np.split(class_dict["oil"],
                                                  [int(len(class_dict["oil"]) * split[0]),
                                                   int(len(class_dict["oil"]) * (split[0] + split[1]))])
        train_watercolor, test_watercolor, valid_watercolor = np.split(class_dict["watercolor"],
                                                                       [int(len(class_dict["watercolor"]) * split[0]),
                                                                        int(len(class_dict["watercolor"]) * (split[0] + split[1]))])
        
        train_ids = {"oil": train_oil, "watercolor": train_watercolor}
        test_ids = {"oil": test_oil, "watercolor": test_watercolor}
        valid_ids = {"oil": valid_oil, "watercolor": valid_watercolor}

        train_images, train_labels, test_images, test_labels, valid_images, valid_labels, camera_images, camera_labels, combine_images, combine_labels = (
                [] for i in range(10)) # Initialize lists
        
        gray_img_pro = ImageProcessor()
        gray_img_pro.target_size = img_pro.target_size
        gray_img_pro.color_space = "L"
        gray_img_pro.resample_filter = img_pro.resample_filter

        # train_images
        for material in train_ids:
            for n, img_path in enumerate(train_ids[material]):
                img = Image.open(img_path)
                img = img_pro.PrepareImage(img)
                if gray_channel:
                    gray_img = gray_img_pro.PrepareImage(img)

                new_img = np.asarray(img)
                if edge_channel: # Converting PIL image in to CV2 image format
                    cv_img = new_img[:, :, ::-1].copy()
                    cv_img = EdgeDetection(cv_img)
                    cv_img = np.expand_dims(cv_img, axis=-1)
                if gray_channel:
                    gray_img = np.asarray(gray_img)
                    gray_img = np.expand_dims(gray_img, axis=-1) # Make array 3D. (Input: 230,230 - Output: 230,230,1)
                    new_img = np.concatenate((new_img, gray_img), axis=2) # Adding gray channel to array (Output: 230,230,4)
                if edge_channel:
                        new_img = np.concatenate((new_img, cv_img), axis=2) # Adding gray channel to array (Output: 230,230,5)

                train_images.append(new_img)
                train_labels.append(class_names.index(material))
                if random_effect:
                    img_pro.random_seed = n

                    new_img = img_pro.RandomEffect(img, "Rotate")
                    if gray_channel:
                        gray_img = gray_img_pro.PrepareImage(new_img)
                    
                    new_img = np.asarray(new_img)
                    if edge_channel:
                        cv_image = new_img[:, :, ::-1].copy()
                        cv_image = EdgeDetection(cv_image)
                        cv_image = np.expand_dims(cv_image, axis=-1)
                    if gray_channel:
                        gray_img = np.asarray(gray_img)
                        gray_img = np.expand_dims(gray_img, axis=-1)
                        new_img = np.concatenate((new_img, gray_img), axis=2)
                    if edge_channel:
                        new_img = np.concatenate((new_img, cv_img), axis=2)
                    train_images.append(new_img)
                    train_labels.append(class_names.index(material))

                    new_img = img_pro.RandomEffect(img, "Flip")
                    if gray_channel:
                        gray_img = gray_img_pro.PrepareImage(new_img)
                    
                    new_img = np.asarray(new_img)
                    if edge_channel:
                        cv_image = new_img[:, :, ::-1].copy()
                        cv_image = EdgeDetection(cv_image)
                        cv_image = np.expand_dims(cv_image, axis=-1)
                    if gray_channel:
                        gray_img = np.asarray(gray_img)
                        gray_img = np.expand_dims(gray_img, axis=-1)
                        new_img = np.concatenate((new_img, gray_img), axis=2)
                    if edge_channel:
                        new_img = np.concatenate((new_img, cv_img), axis=2)
                    train_images.append(np.asarray(new_img))
                    train_labels.append(class_names.index(material))

                    new_img = img_pro.RandomEffect(img, ["Rotate", "Flip"])
                    if gray_channel:
                        gray_img = gray_img_pro.PrepareImage(new_img)
                    
                    new_img = np.asarray(new_img)
                    if edge_channel:
                        cv_image = new_img[:, :, ::-1].copy()
                        cv_image = EdgeDetection(cv_image)
                        cv_image = np.expand_dims(cv_image, axis=-1)
                    if gray_channel:
                        gray_img = np.asarray(gray_img)
                        gray_img = np.expand_dims(gray_img, axis=-1)
                        new_img = np.concatenate((new_img, gray_img), axis=2)
                    if edge_channel:
                        new_img = np.concatenate((new_img, cv_img), axis=2)
                    train_images.append(new_img)
                    train_labels.append(class_names.index(material))

        # test_images
        for material in test_ids:
            for n, img_path in enumerate(test_ids[material]):
                img = Image.open(img_path)
                img = img_pro.PrepareImage(img)
                if gray_channel:
                    gray_img = gray_img_pro.PrepareImage(img)

                new_img = np.asarray(img)
                if edge_channel:
                    cv_img = new_img[:, :, ::-1].copy()
                    cv_img = EdgeDetection(cv_img)
                    cv_img = np.expand_dims(cv_img, axis=-1)
                if gray_channel:
                    gray_img = np.asarray(gray_img)
                    gray_img = np.expand_dims(gray_img, axis=-1)
                    new_img = np.concatenate((new_img, gray_img), axis=2)
                if edge_channel:
                        new_img = np.concatenate((new_img, cv_img), axis=2)

                test_images.append(new_img)
                test_labels.append(class_names.index(material))

        # valid_images
        for material in valid_ids:
            for n, img_path in enumerate(valid_ids[material]):
                img = Image.open(img_path)
                img = img_pro.PrepareImage(img)
                if gray_channel:
                    gray_img = gray_img_pro.PrepareImage(img)

                new_img = np.asarray(img)
                if edge_channel:
                    cv_image = new_img[:, :, ::-1].copy()
                    cv_image = EdgeDetection(cv_image)
                    cv_image = np.expand_dims(cv_image, axis=-1)
                if gray_channel:
                    gray_img = np.asarray(gray_img)
                    gray_img = np.expand_dims(gray_img, axis=-1)
                    new_img = np.concatenate((new_img, gray_img), axis=2)
                if edge_channel:
                    new_img = np.concatenate((new_img, cv_img), axis=2)

                valid_images.append(new_img)
                valid_labels.append(class_names.index(material))

    if not build:
        return (train_images, train_labels), (test_images, test_labels), (valid_images, valid_labels), (camera_images, camera_labels)

    if not disable_cache:
        if not using_cache:
            ArtMediumData = {
                "train": (train_images, train_labels),
                "test": (test_images, test_labels),
                "valid": (valid_images, valid_labels),
                "camera": (camera_images, camera_labels),
            }

            # Save values to speed up future calls (Creates very large files. 9GB+ when using Edge and Random)
            with open(filepath, "wb") as f:
                pickle.dump(ArtMediumData, f)

    # Combination of all data not used for training
    combine_images = test_images + valid_images #+ camera_images
    combine_labels = test_labels + valid_labels #+ camera_labels
    
    
    (train_images, train_labels), (test_images, test_labels), (valid_images, valid_labels), (
        camera_images, camera_labels), (combine_images, combine_labels) = BuildDataset(train_images, train_labels, 
                                                                                       test_images, test_labels, 
                                                                                       valid_images, valid_labels, 
                                                                                       camera_images, camera_labels, 
                                                                                       combine_images, combine_labels)
    
    return (train_images, train_labels), (test_images, test_labels), (valid_images, valid_labels), (
        camera_images, camera_labels), (combine_images, combine_labels)


def BuildDataset(train_images, train_labels, test_images, test_labels, valid_images, valid_labels, camera_images, camera_labels, combine_images, combine_labels):

    train_images = np.array(train_images)
    train_labels = np.array(train_labels, dtype=int)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels, dtype=int)

    valid_images = np.array(valid_images)
    valid_labels = np.array(valid_labels, dtype=int)

    camera_images = np.array(camera_images)
    camera_labels = np.array(camera_labels, dtype=int)

    combine_images = np.array(combine_images)
    combine_labels = np.array(combine_labels, dtype=int)

    return (train_images, train_labels), (test_images, test_labels), (valid_images, valid_labels), (camera_images, camera_labels), (combine_images, combine_labels)


if __name__ == "__main__":
    img_pro = ImageProcessor()
    img_pro.target_size = (230, 230)
    img_pro.color_space = "RGB" # Only use 3 channel color spaces. (RGB, LAB, HSV)
    img_pro.resample_filter = Image.Resampling.LANCZOS # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters

    start = time.perf_counter()
    (train_images, train_labels), (test_images, test_labels
    ), (valid_images, valid_labels), (camera_images, camera_labels
    ), (combine_images, combine_labels) = ArtMediumDataset(
                                            img_pro,
                                            train_hold=None, # Currently unused
                                            use_cache=False,  
                                            split=(80, 10, 10), # (Training, Test, Valid)
                                            random_effect=True, # Only applies to training data
                                            gray_channel=True,
                                            edge_channel=True,
                                            build=True, # Should always be True (placeholder for adding datasets together)
                                            disable_cache=True, #Change to False to save data for faster loading (Creates large files)
                                            debug=True)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Parsing took: {elapsed:.2f} seconds")
