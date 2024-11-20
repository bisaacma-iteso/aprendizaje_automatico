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
from cuml.manifold import TSNE
from cuml.model_selection import train_test_split
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE as cpu_TSNE

import cudf
import pandas as pd
import os
import cupy as cp
import gc

import sys

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
             debug=False,
             use_grayscale_only=False
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

    if use_grayscale_only:
        return gray_images

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

        # Gaussian filter
        masked_image_scaled = binary_image
        if use_gaussian_mask:
            masked_image_scaled, mask = apply_gaussian_filter(binary_image, gaussian_sigma)

        new_dataset[i] = masked_image_scaled
        
        # Log filter
        # log_filtered_image = apply_log_filter(median_filtered_image)

        if (i == random_sample_id) and debug:
            contour_image = find_contours(binary_image)
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
                 debug=False,
                 use_grayscale_only=False):
        # Store preprocessing parameters
        self.clahe_clipLimit = clahe_clipLimit
        self.clahe_tileGridSize = clahe_tileGridSize
        self.sharpen_filter = sharpen_filter
        self.use_gaussian_mask = use_gaussian_mask
        self.gaussian_sigma = gaussian_sigma
        self.extra_filter = extra_filter
        self.random_sample_id = random_sample_id
        self.debug = debug
        self.use_grayscale_only = use_grayscale_only

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
            debug=self.debug,
            use_grayscale_only=self.use_grayscale_only
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
gaussian_mask_range = [True, False]
gaussian_sigma_range = list(range(1, 10))
#gaussian_sigma_range = [6]
extra_filtering_range = ['median', 'log']

knn_range = [30]
knn_range = list(range(30,100))
pca_range = [47]
pca_range = list(range(55,80))
#pca_range = [50]
svm_c_range = [1]  # Range of SVM regularization parameter (C)
svm_kernel_range = ['linear', 'rbf']  # SVM kernels
svm_kernel_range = ['poly']  # SVM kernels
svm_poly_degree_range = [4, 5, 6]
umap_range = list(range(47, 70))  # Different values of UMAP components you want to test
umap_neighbors_range = list(range(5, 51))
umap_metric_range = ['euclidean', 'cosine', 'manhattan', 'hamming']
umap_min_dist_range = np.arange(0.0, 1.1, 0.1).round(2).tolist()
umap_min_dist_range = [1]


# Logistic regression hyperparameter grid
log_c_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]  # Regularization parameter range
log_max_iter_range = [100, 1000, 10000]  # Maximum iterations

# Hyperparameters for t-SNE and KNN
tsne_perplexity_range = [5, 10, 20, 30]  # t-SNE perplexity values
tsne_learning_rate_range = [1, 10, 100, 1000]  # t-SNE learning rates
tsne_n_iter_range = [500, 1000, 10000]  # t-SNE max iterations
tsne_num_components = list(range(1, 4))

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
        ('pca', cuPCA()),  # Use cuML's UMAP for GPU acceleration
        ('svm', cuSVC())   # Use cuSVC for GPU acceleration
    ])

# Logistic regression pipeline
log_pipeline = Pipeline(
    steps=[
        ('pca',     cuPCA()),  # UMAP for dimensionality reduction
        ('log_reg', cuLogisticRegression())  # Logistic Regression using cuML
    ]
)

# Create a pipeline with t-SNE and SVM
tsne_svm_pipeline = Pipeline(
    steps=[
        ('svm', cuSVC())   # Use cuKNN for GPU acceleration
    ]
)

# Manually implement grid search on the GPU (since GridSearchCV isn't supported in cuML)
best_score = -np.inf
best_params = {}

cv_to_run = "PCA_AND_SVM"
if len(sys.argv) > 1:
    cv_to_run = sys.argv[1]

# Iterate through possible values of PCA components and KNN neighbors
if cv_to_run == "PCA_AND_KNN":
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
if cv_to_run == "PCA_AND_SVM":
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
                    for C in svm_c_range:
                        for kernel in svm_kernel_range:
                            for degree in svm_poly_degree_range:
                                # Update the PCA and SVM models with the current hyperparameters
                                svm_pipeline.set_params(pca__n_components=n_components,
                                                        svm__C=C,
                                                        svm__kernel=kernel,
                                                        svm__degree=degree)

                                # Fit the model on the GPU
                                svm_pipeline.fit(X_pca_train, Y_train)

                                # Get the accuracy score on the test set
                                score = svm_pipeline.score(X_pca_test, Y_test)

                                current_params = {'preprocessing__use_gaussian_mask' : n_use_gaussian,
                                            'preprocessing__gaussian_sigma' : n_gaussian_sigma,
                                            'preprocessing__extra_filter' : n_filter,
                                            'pca__n_components': n_components,
                                            'svm__C': C,
                                            'svm__kernel': kernel,
                                            'svm__degree' : degree
                                }


                                print(f'Finished {current_params} | score:{score}')

                                # Store the best hyperparameters and score
                                if score > best_score:
                                    best_score = score
                                    best_params = current_params
                                
                                cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                                gc.collect()  # Run garbage collection to clear Python memory

# Iterate through possible values of UMAP and SVM hyperparameters
if cv_to_run == "UMAP_AND_SVM":

    transformer = PreprocessingTransformer(gaussian_sigma=6,
                                        use_gaussian_mask=True,
                                        extra_filter='median')
    X_pca_train = transformer.transform(ds_train_images)
    X_pca_test  = transformer.transform(ds_test_images)
    
    
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
                    for n_components in umap_range:
                        for n_umap_neighbors in umap_neighbors_range:
                            for metric_umap in umap_metric_range:
                                for mindist_umap in umap_min_dist_range:
                                    for C in svm_c_range:
                                        for kernel in svm_kernel_range:
                                            for degree in svm_poly_degree_range:

                                                # Step 1: Run UMAP transformation separately
                                                umap_transformer = UMAP(n_components=n_components)
                                                X_train_umap = umap_transformer.fit_transform(X_pca_train)
                                                X_test_umap = umap_transformer.transform(X_pca_test)
                                                
                                                # Update the PCA and SVM models with the current hyperparameters
                                                # Step 2: Train SVM
                                                svm_model = cuSVC(C=C, kernel=kernel, degree=degree)
                                                svm_model.fit(X_train_umap, Y_train)

                                                # Get the accuracy score on the test set
                                                score = svm_model.score(X_test_umap, Y_test)

                                                current_params = {'preprocessing__use_gaussian_mask' : n_use_gaussian,
                                                            'preprocessing__gaussian_sigma' : n_gaussian_sigma,
                                                            'preprocessing__extra_filter' : n_filter,
                                                            'umap__n_components': n_components,
                                                            'umap__n_neighbors' : n_umap_neighbors,
                                                            'umap__metric' : metric_umap,
                                                            'umap__min_dist' : mindist_umap,
                                                            'svm__C': C,
                                                            'svm__kernel': kernel,
                                                            'svm__degree' : degree
                                                }

                                                print(f'Finished {current_params}\n\tscore:{score}\n\tbest_score_so_far:{best_score}')

                                                # Store the best hyperparameters and score
                                                if score > best_score:
                                                    best_score = score
                                                    best_params = current_params
                                                
                                                #cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                                                gc.collect()  # Run garbage collection to clear Python memory

# Try with logistic regression
# Iterate through possible values of PCA components and KNN neighbors
if cv_to_run == "PCA_AND_LOGISTIC":
    for n_use_gaussian in gaussian_mask_range:
        for n_gaussian_sigma in gaussian_sigma_range:
            for n_filter in extra_filtering_range:
                # Preprocess data
                transformer = PreprocessingTransformer(
                    gaussian_sigma=n_gaussian_sigma,
                    use_gaussian_mask=n_use_gaussian,
                    extra_filter=n_filter
                )
                X_train_processed = transformer.transform(ds_train_images)
                X_test_processed = transformer.transform(ds_test_images)

                for n_components in pca_range:
                    for c in log_c_range:
                        for max_iter in log_max_iter_range:
                            # Update pipeline hyperparameters
                            log_pipeline.set_params(
                                pca__n_components=n_components,
                                log_reg__C=c,
                                log_reg__max_iter=max_iter
                            )

                            # Fit the model
                            log_pipeline.fit(X_train_processed, Y_train)

                            # Evaluate on the test set
                            score = log_pipeline.score(X_test_processed, Y_test)

                            current_params = {
                                'preprocessing__use_gaussian_mask': n_use_gaussian,
                                'preprocessing__gaussian_sigma': n_gaussian_sigma,
                                'preprocessing__extra_filter': n_filter,
                                'pca__n_components': n_components,
                                'log_reg__C': c,
                                'log_reg__max_iter': max_iter
                            }

                            # Update the best parameters and score
                            if score > best_score:
                                best_score = score
                                best_params = current_params

                            print(f'Finished {current_params} | Score: {score}')

if cv_to_run == "TSNE_AND_SVM":
    # Iterate through hyperparameters
    for n_use_gaussian in gaussian_mask_range:
        for index, n_gaussian_sigma in enumerate(gaussian_sigma_range):
            for n_filter in extra_filtering_range:
                # Preprocess data
                transformer = PreprocessingTransformer(gaussian_sigma=n_gaussian_sigma,
                                                       use_gaussian_mask=n_use_gaussian,
                                                       extra_filter=n_filter)
                X_pre_train = transformer.transform(ds_train_images)
                X_pre_test  = transformer.transform(ds_test_images)
                for perplexity in tsne_perplexity_range:
                    for learning_rate in tsne_learning_rate_range:
                        for n_iter in tsne_n_iter_range:
                            for C in svm_c_range:
                                for kernel in svm_kernel_range:
                                    for degree in svm_poly_degree_range:
                                        for n_components in tsne_num_components:
                                            # Update the t-SNE and KNN models with the current hyperparameters
                                            tsne_svm_pipeline.set_params(
                                                svm__C=C,
                                                svm__kernel=kernel,
                                                svm__degree=degree
                                            )

                                            tsne_model = cpu_TSNE(perplexity=perplexity,
                                                                  learning_rate=learning_rate,
                                                                  n_iter=n_iter,
                                                                  n_components=n_components)
                                            X_pre_train = tsne_model.fit_transform(X_pre_train.get())
                                            X_pre_test  = tsne_model.fit_transform(X_pre_test.get())
                                            # Fit the pipeline on the training data
                                            
                                            X_pre_train = cudf.DataFrame.from_pandas(pd.DataFrame(X_pre_train)).to_cupy()
                                            X_pre_test = cudf.DataFrame.from_pandas(pd.DataFrame(X_pre_test)).to_cupy()

                                            tsne_svm_pipeline.fit(X_pre_train, Y_train)

                                            # Evaluate on the test data
                                            score = tsne_svm_pipeline.score(X_pre_test, Y_test)

                                            current_params = {'preprocessing__use_gaussian_mask' : n_use_gaussian,
                                                    'preprocessing__gaussian_sigma' : n_gaussian_sigma,
                                                    'preprocessing__extra_filter' : n_filter,
                                                    'tsne__n_components' : n_components,
                                                    'tsne__perplexity': perplexity,
                                                    'tsne__learning_rate': learning_rate,
                                                    'tsne__n_iter': n_iter,
                                                    'svm__C': C,
                                                    'svm__kernel': kernel,
                                                    'svm__degree' : degree}

                                            # Save the best parameters and score
                                            if score > best_score:
                                                best_score = score
                                                best_params = current_params

                                            print(f'Finished {current_params} | score:{score}')

if cv_to_run == "PREPROCESSING":
    for grayscale_only in [False, True]:
        for n_use_gaussian in gaussian_mask_range:
            for index, n_gaussian_sigma in enumerate(gaussian_sigma_range):
                for n_filter in extra_filtering_range:
                    # Preprocess data
                    transformer = PreprocessingTransformer(gaussian_sigma=n_gaussian_sigma,
                                                        use_gaussian_mask=n_use_gaussian,
                                                        extra_filter=n_filter,
                                                        use_grayscale_only=grayscale_only)
                    X_pca_train = transformer.transform(ds_train_images)
                    X_pca_test  = transformer.transform(ds_test_images)
                    for n_components in [57]:
                        for C in [1]:
                            for kernel in ['poly']:
                                for degree in [6]:
                                    # Update the PCA and SVM models with the current hyperparameters
                                    svm_pipeline.set_params(pca__n_components=n_components,
                                                            svm__C=C,
                                                            svm__kernel=kernel,
                                                            svm__degree=degree)

                                    # Fit the model on the GPU
                                    svm_pipeline.fit(X_pca_train, Y_train)

                                    # Get the accuracy score on the test set
                                    score = svm_pipeline.score(X_pca_test, Y_test)

                                    current_params = {
                                                'preprocessing__use_grayscale_only' : grayscale_only,
                                                'preprocessing__use_gaussian_mask' : n_use_gaussian,
                                                'preprocessing__gaussian_sigma' : n_gaussian_sigma,
                                                'preprocessing__extra_filter' : n_filter,
                                                'pca__n_components': n_components,
                                                'svm__C': C,
                                                'svm__kernel': kernel,
                                                'svm__degree' : degree
                                    }


                                    print(f'Finished {current_params} | score:{score}')

                                    # Store the best hyperparameters and score
                                    if score > best_score:
                                        best_score = score
                                        best_params = current_params
                                    
                                    cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                                    gc.collect()  # Run garbage collection to clear Python memory

# Output the best hyperparameters and score
print(f"{cv_to_run} | Best parameters: {best_params}")
print(f"{cv_to_run} | Best score: {best_score}")

# With stupid pre-processing
#Best parameters: {'pca__n_components': 49, 'knn__n_neighbors': 31}
#Best score: 0.5820144414901733

#Best parameters: {'pca__n_components': 48, 'knn__n_neighbors': 31}
#Best score: 0.6881146430969238

#Finished {'preprocessing__gaussian_sigma': 6, 'preprocessing__extra_filter': 'median', 'pca__n_components': 49, 'knn__n_neighbors': 31} | score:0.7254148721694946

#Best parameters: {'preprocessing__gaussian_sigma': 6, 'preprocessing__extra_filter': 'median', 'pca__n_components': 47, 'knn__n_neighbors': 30}
#Best score: 0.7265673279762268


#Best parameters: {'preprocessing__use_gaussian_mask': True, 'preprocessing__gaussian_sigma': 6, 'preprocessing__extra_filter': 'median', 'pca__n_components': 57, 'knn__n_neighbors': 19}
#Best score: 0.732291042804718

# Best so far
#PCA_AND_SVM | Best parameters: {'preprocessing__use_gaussian_mask': True, 'preprocessing__gaussian_sigma': 6, 'preprocessing__extra_filter': 'median', 'pca__n_components': 57, 'svm__C': 1, 'svm__kernel': 'poly', 'svm__degree': 6}
#PCA_AND_SVM | Best score: 0.7990934252738953