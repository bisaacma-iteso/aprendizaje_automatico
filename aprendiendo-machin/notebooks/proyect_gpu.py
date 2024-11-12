import huggingface_hub
from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import cv2

from skimage import color, exposure, filters, util, morphology
from skimage import img_as_ubyte, img_as_float
from skimage.filters import rank
from skimage.filters import gaussian
from sklearn.preprocessing import FunctionTransformer

from cuml.decomposition import PCA as cuPCA
from cuml.preprocessing import minmax_scale as cuml_minmax_scale
from cuml.svm import LinearSVC as cuLinearSVC
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.svm import SVC as cuSVC
from cuml.manifold import UMAP
from cuml.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import cudf
import pandas as pd
import os
import cupy as cp
import gc

from sklearn.base import BaseEstimator, TransformerMixin

# Import cropped-digits dataset
dataset_cropped_digits_raw = load_dataset("ufldl-stanford/svhn",
                                          "cropped_digits",
                                          cache_dir='../data/svhn/')

# Get dataset subsets
ds_train = dataset_cropped_digits_raw["train"]
ds_test = dataset_cropped_digits_raw["test"]

ds_train_images = np.array(ds_train["image"])
ds_test_images = np.array(ds_test["image"])

def apply_gaussian_filter(image, sigma=6):
    # Create a Gaussian mask that varies only horizontally based on distance from the center
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    center_x = width / 2

    # Create a Gaussian mask
    # sigma = Standard deviation for the Gaussian function
    mask = np.exp(-((x - center_x)**2) / (2 * sigma**2))

    # Apply the Gaussian mask to the image
    masked_image = image * mask

    # Scale the masked image to 0-255
    masked_image_scaled = (masked_image - np.min(masked_image)) / (np.max(masked_image) - np.min(masked_image)) * 255
    masked_image_scaled = masked_image_scaled.astype(np.uint8)

    return masked_image_scaled, mask

def remove_borders(image, reduce_factor=0.1):
    # Define the border width as 10% of the image size
    border_size = int(32 * reduce_factor)

    blackout_image = image.copy()
    # Set the border regions to black (0)
    # Top border
    blackout_image[:border_size, :] = 0
    # Bottom border
    blackout_image[-border_size:, :] = 0
    # Left border
    blackout_image[:, :border_size] = 0
    # Right border
    blackout_image[:, -border_size:] = 0

    return blackout_image

def find_contours(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Create a blank image to draw contours on
    contour_only_image = np.zeros_like(image)
    cv2.drawContours(contour_only_image, contours, -1, 255, 1)  # Draw contours in white (255)
    return contour_only_image

def sharpen_image(image, filter=(5,5)):
    # Sharpen image
    blurred = cv2.GaussianBlur(image, filter, 0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

def apply_median_filter(image):
    # Step 2: Reduce the impact of outliers using median filter
    selem = morphology.disk(2)
    return filters.rank.median(img_as_ubyte(image), selem)

def apply_log_filter(image):
    # Create a FunctionTransformer for the log transformation
    log_transformer = FunctionTransformer(np.log1p, validate=True)

    # Apply the log transformation
    transformed_image = log_transformer.fit_transform(image)

    # Rescale the transformed image back to [0, 255]
    transformed_image = (transformed_image / np.max(transformed_image)) * 255
    return transformed_image.astype(np.uint8)

def pipeline(ds,
             labels_ds,
             clahe_clipLimit=2.0,
             clahe_tileGridSize=(3,3),
             sharpen_filter=(5,5),
             use_gaussian_mask=True,
             gaussian_sigma=6,
             extra_filter=None,
             random_sample_id= None,
             debug=False
             ):
    """ 
    Assumes input dataset are N samples of RGB images, 32x32 (32, 32, 3) shape.
    Input must be a np-array with images only.
    Transforms from RGB (3-channels) to a single grayscale channel
    Applies Gaussian weighting to remove distractions in borders
    Actual digit is centered

    Returns modified dataset
    """
    # Initialize an array to hold the grayscale images
    new_dataset = np.empty((len(ds), 32, 32), dtype=np.uint8)

    if (random_sample_id is None) and debug:
        random_sample_id = random.randint(0, len(ds))
    
    if debug:
        original_random_image = ds[random_sample_id]
        original_random_label = labels_ds[random_sample_id]

    # Step 1: Convert images to grayscale
    gray_images = (color.rgb2gray(ds) * 255).astype(np.uint8)

    # Iterate per-image
    total_images_processed = 0
    for i in range(0, len(gray_images)):
        
        total_images_processed +=1
        gray_image = gray_images[i]

        # Sharpening
        sharpened_image = sharpen_image(gray_image, sharpen_filter)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize)
        clahe_image = clahe.apply(sharpened_image)

        # Median filter
        median_filtered_image = apply_median_filter(clahe_image)

        # Log filter
        log_filtered_image = apply_log_filter(clahe_image)

        image_for_binary = clahe_image
        if (extra_filter=="log"):
            image_for_binary = log_filtered_image
        if (extra_filter=="median"):
            image_for_binary = median_filtered_image

        # Use binary threshold to reduce everything to 2 colors
        _, binary_image = cv2.threshold(image_for_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #print(f'mean_binary: {np.mean(binary_image)}')

        # Get contours
        contour_image = find_contours(binary_image)

        # Gaussian filter
        masked_image_scaled = binary_image
        if use_gaussian_mask:
            masked_image_scaled, mask = apply_gaussian_filter(binary_image, gaussian_sigma)

        new_dataset[i] = masked_image_scaled
        
        # Log filter
        # log_filtered_image = apply_log_filter(median_filtered_image)

        if (i == random_sample_id) and debug:
            # List of image data and titles for each row
            images = [
                (original_random_image, gray_image),       # Row 1 images
                (sharpened_image, clahe_image),             # Row 2 images
                (log_filtered_image, median_filtered_image),          # Row 3 images
                (binary_image, contour_image),                     # Row 4 images
                (mask, masked_image_scaled)                # Row 5 image (second image is None)
            ]

            # List of titles for each row
            titles = [
                ("Imagen Original", "Escala de Grises"),
                ("Sharpening", "CLAHE"),
                ("CLAHE + Log", "CLAHE + Median"),
                ("Binary Dynamic Threshold", "Contornos"),
                ("Máscara Gaussiana", "Imagen enmascarada")  # No second title for this row
            ]

            # Loop over the rows to create and save individual images for each row
            for i, (row_images, row_titles) in enumerate(zip(images, titles)):
                # Create a figure for the current row
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 images per row

                primer_paso = i*2
                segundo_paso = (i*2)+1
                #fig.suptitle(f'Pasos {primer_paso} y {segundo_paso}', fontsize=16)

                # Plot the first image in the row
                axes[0].imshow(row_images[0], cmap='gray')
                axes[0].set_title(row_titles[0])
                axes[0].axis('off')

                # If there is a second image, plot it
                if row_images[1] is not None:
                    axes[1].imshow(row_images[1], cmap='gray')
                    axes[1].set_title(row_titles[1])
                    axes[1].axis('off')
                else:
                    axes[1].axis('off')  # Hide the second axis if no second image

                # Save the image for the current row
                plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust spacing if necessary

                plt.savefig(f'row_{i+1}_images.png', dpi=300, bbox_inches='tight')  # Save each row's image
                #plt.close()  # Close the figure to free up memory

    return new_dataset


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clahe_clipLimit=2.0,
                 clahe_tileGridSize=(3, 3),
                 sharpen_filter=(5, 5),
                 use_gaussian_mask=True,
                 gaussian_sigma=6,
                 extra_filter=None,
                 random_sample_id=None,
                 debug=False):
        # Store preprocessing parameters
        self.clahe_clipLimit = clahe_clipLimit
        self.clahe_tileGridSize = clahe_tileGridSize
        self.sharpen_filter = sharpen_filter
        self.use_gaussian_mask = use_gaussian_mask
        self.gaussian_sigma = gaussian_sigma
        self.extra_filter = extra_filter
        self.random_sample_id = random_sample_id
        self.debug = debug

    def fit(self, X, y=None):
        # This method can stay empty since we don't need fitting for preprocessing
        return self

    def transform(self, X, y=None):

        # CPU-only processed data
        processed_data = pipeline(
            ds=X, 
            labels_ds=y, 
            clahe_clipLimit=self.clahe_clipLimit, 
            clahe_tileGridSize=self.clahe_tileGridSize, 
            sharpen_filter=self.sharpen_filter,
            use_gaussian_mask=self.use_gaussian_mask,
            gaussian_sigma=self.gaussian_sigma, 
            extra_filter=self.extra_filter, 
            random_sample_id=self.random_sample_id,
            debug=self.debug
        )

        # Flatten data
        processed_data = processed_data.reshape(processed_data.shape[0], -1)

        # Scale the data on GPU
        processed_data = cuml_minmax_scale(processed_data, feature_range=(0, 1))

        # Convert to cudf DataFrame on GPU
        processed_data = cudf.DataFrame.from_pandas(pd.DataFrame(processed_data)).to_cupy()
        return processed_data


# Test
#transformer = PreprocessingTransformer(gaussian_sigma=6,
#                                                    extra_filter='median')
#X_pca_train = transformer.transform(np.expand_dims(ds_train_images[0], axis=0))

# Feed to pipeline
Y_train = ds_train["label"]
Y_test = ds_test["label"]

# Convert to cudf DataFrame on GPU
Y_train = cudf.DataFrame.from_pandas(pd.DataFrame(Y_train)).to_cupy().ravel()
Y_test = cudf.DataFrame.from_pandas(pd.DataFrame(Y_test)).to_cupy().ravel()

# Create GPU-accelerated versions of the models

# Hyperparameter range for transformation
gaussian_mask_range = [True]
gaussian_sigma_range = list(range(1, 10))
extra_filtering_range = ['median']

knn_range = [30]
pca_range = [47]
svm_c_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]  # Range of SVM regularization parameter (C)
svm_kernel_range = ['linear', 'rbf']  # SVM kernels
umap_range = list(range(35,38))  # Different values of UMAP components you want to test

# Create GPU-accelerated PCA, KNN, and SVC pipelines
knn_pipeline = Pipeline(
    steps=[
        ('pca', cuPCA()),  # Use cuPCA for GPU acceleration
        ('knn', cuKNN())   # Use cuKNN for GPU acceleration
    ])

""" svm_pipeline = Pipeline(
    steps=[
        ('pca', cuPCA()),  # Use cuPCA for GPU acceleration
        ('svm', cuSVC())   # Use cuSVC for GPU acceleration
    ]) """

svm_pipeline = Pipeline(
    steps=[
        ('umap', UMAP(n_components=3)),  # Use cuML's UMAP for GPU acceleration
        ('svm', cuSVC())   # Use cuSVC for GPU acceleration
    ])

# Manually implement grid search on the GPU (since GridSearchCV isn't supported in cuML)
best_score = -np.inf
best_params = {}

# Iterate through possible values of PCA components and KNN neighbors
if True:
    for n_use_gaussian in gaussian_mask_range:
        for index, n_gaussian_sigma in enumerate(gaussian_sigma_range):
            for n_filter in extra_filtering_range:
                # Preprocess data
                transformer = PreprocessingTransformer(gaussian_sigma=n_gaussian_sigma,
                                                       use_gaussian_mask=n_use_gaussian,
                                                       extra_filter=n_filter)
                X_pca_train = transformer.transform(ds_train_images)
                X_pca_test  = transformer.transform(ds_test_images)
                for n_components in pca_range:
                    for n_neighbors in knn_range:
                        
                        # Update the PCA and KNN models with the current hyperparameters
                        knn_pipeline.set_params(pca__n_components=n_components,
                                                knn__n_neighbors=n_neighbors)

                        # Fit the model on the GPU
                        knn_pipeline.fit(X_pca_train, Y_train)

                        # Get the accuracy score on the test set
                        score = knn_pipeline.score(X_pca_test, Y_test)

                        current_params = {'preprocessing__use_gaussian_mask' : n_use_gaussian,
                                          'preprocessing__gaussian_sigma' : n_gaussian_sigma,
                                          'preprocessing__extra_filter' : n_filter,
                                          'pca__n_components': n_components,
                                          'knn__n_neighbors': n_neighbors}

                        # Store the best hyperparameters and score
                        if score > best_score:
                            best_score = score
                            best_params = current_params
                        
                        print(f'Finished {current_params} | score:{score}')
                        #cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                        #gc.collect()  # Run garbage collection to clear Python memory

# Iterate through possible values of PCA components and SVM hyperparameters (C and kernel)
if False:
    for n_components in pca_range:
        for C in svm_c_range:
            for kernel in svm_kernel_range:
                # Update the PCA and SVM models with the current hyperparameters
                svm_pipeline.set_params(pca__n_components=n_components, svm__C=C, svm__kernel=kernel)

                # Fit the model on the GPU
                svm_pipeline.fit(X_pca_train, Y_train)

                # Get the accuracy score on the test set
                score = svm_pipeline.score(X_pca_test, Y_test)

                print(f'Finished PCA components:{n_components}, svm_c:{C}, svm_kernel:{kernel} score:{score}')

                # Store the best hyperparameters and score
                if score > best_score:
                    best_score = score
                    best_params = {'pca__n_components': n_components, 'svm__C': C, 'svm__kernel': kernel}
                
                cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                gc.collect()  # Run garbage collection to clear Python memory

if False:
    for n_gaussian_sigma in gaussian_sigma_range:
        for n_filter in extra_filtering_range:
            # Preprocess data
            transformer = PreprocessingTransformer(gaussian_sigma=n_gaussian_sigma,
                                                    extra_filter=n_filter)
            X_pca_train = transformer.transform(ds_train_images)
            X_pca_test  = transformer.transform(ds_test_images)
            for n_components in umap_range:
                for C in svm_c_range:
                    for kernel in svm_kernel_range:

                        svm_pipeline = Pipeline(steps=[
                                                    ('umap', UMAP(n_components=3)),  # Use cuML's UMAP for GPU acceleration
                                                    ('svm', cuSVC())   # Use cuSVC for GPU acceleration
                                                ])
                        
                        # Update the UMAP and SVM models with the current hyperparameters
                        svm_pipeline.set_params(umap__n_components=n_components,
                                                svm__C=C,
                                                svm__kernel=kernel)

                        # Fit the model on the GPU
                        svm_pipeline.fit(X_pca_train, Y_train)

                        # Get the accuracy score on the test set
                        score = svm_pipeline.score(X_pca_test, Y_test)

                        current_params = {'preprocessing__gaussian_sigma' : n_gaussian_sigma,
                                        'preprocessing__extra_filter' : n_filter,
                                        'umap__n_components': n_components,
                                        'svm__C': C,
                                        'svm__kernel' : kernel}

                        # Store the best hyperparameters and score
                        if score > best_score:
                            best_score = score
                            best_params = current_params
                        
                        print(f'Finished {current_params} | score:{score}')
                        cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                        gc.collect()  # Run garbage collection to clear Python memory
                        del svm_pipeline

# Output the best hyperparameters and score
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

# With stupid pre-processing
#Best parameters: {'pca__n_components': 49, 'knn__n_neighbors': 31}
#Best score: 0.5820144414901733

#Best parameters: {'pca__n_components': 48, 'knn__n_neighbors': 31}
#Best score: 0.6881146430969238

#Finished {'preprocessing__gaussian_sigma': 6, 'preprocessing__extra_filter': 'median', 'pca__n_components': 49, 'knn__n_neighbors': 31} | score:0.7254148721694946

#Best parameters: {'preprocessing__gaussian_sigma': 6, 'preprocessing__extra_filter': 'median', 'pca__n_components': 47, 'knn__n_neighbors': 30}
#Best score: 0.7265673279762268