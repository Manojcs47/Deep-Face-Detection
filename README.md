# Real-Time Face Detection with TensorFlow and VGG16

This repository contains the code for an end-to-end deep learning project that detects human faces in real-time video. The model is built using TensorFlow and leverages transfer learning with the VGG16 architecture to perform both face classification and bounding box regression.

## Project Workflow

The project follows a complete machine learning pipeline from data acquisition to deployment:

1.  **Data Collection**: Images are captured from a webcam using OpenCV.
2.  **Annotation**: The `labelme` tool is used to draw bounding boxes around faces in the collected images, generating JSON files with coordinate data.
3.  **Data Augmentation**: The `Albumentations` library creates a robust training set by applying various transformations to the images and their corresponding bounding boxes. The pipeline includes:
    * `RandomCrop`
    * `HorizontalFlip`
    * `RandomBrightnessContrast`
    * `VerticalFlip`
4.  **Data Pipeline**: The augmented images and labels are loaded into an efficient `tf.data.Dataset` pipeline for training. Images are resized to 120x120 and pixel values are normalized.
5.  **Model Architecture**: A multi-output model is constructed using the TensorFlow Keras Functional API:
    * **Base Model**: A pre-trained VGG16 is used for feature extraction.
    * **Classification Head**: A dense layer branch predicts the presence of a face using a sigmoid activation function.
    * **Regression Head**: A second dense layer branch predicts the four bounding box coordinates, also using a sigmoid activation.
6.  **Custom Training**:
    * A custom model class, `FaceTracker`, is created by inheriting from `tf.keras.Model`.
    * The `train_step` and `test_step` methods are overridden to implement a custom loss logic.
    * The total loss is a combination of `BinaryCrossentropy` for classification and a custom `localization_loss` function for the bounding box regression.
7.  **Deployment**: The trained model is saved as `facetracker.h5` and used in a final script that captures video from a webcam, performs inference, and draws the predicted bounding boxes on the live feed using OpenCV.

## Technologies Used

* **Deep Learning Framework**: TensorFlow, Keras
* **Computer Vision**: OpenCV-Python
* **Data Augmentation**: Albumentations
* **Annotation**: LabelMe
* **Data Handling**: NumPy
* **Visualization**: Matplotlib

## How to Use

1.  **Install Dependencies**:
    ```bash
    pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations
    ```
2.  **Data Collection and Annotation**:
    * Run the cells in section **1.2 Collect Images Using OpenCV** of the notebook to capture images.
    * Use the `!labelme` command in section **1.3 Annotate Images with LabelMe** to annotate your collected images.
3.  **Train the Model**:
    * Manually partition your data into `train`, `test`, and `val` folders.
    * Execute the notebook cells sequentially to run the augmentation pipeline, build the `tf.data` datasets, and train the model. The training history will be saved to a `logs` directory for TensorBoard.
4.  **Run Real-Time Detection**:
    * After training, the model is saved to `facetracker.h5`.
    * Run the final code block in section **11.3 Real Time Detection** to start the webcam feed and see the live face detection.
