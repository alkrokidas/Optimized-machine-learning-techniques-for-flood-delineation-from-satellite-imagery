
# Satellite Climate Intelligence Pipeline: Optimized Deep Learning for Flood Delineation

## Project Overview
As climate risks shift global infrastructure patterns, building robust, automated environmental intelligence networks is critical—particularly for flood-prone, low-lying delta regions like the Netherlands. 

This project delivers an end-to-end data engineering, geospatial preprocessing, and machine learning pipeline designed to delineate flooded regions automatically from raw satellite imagery. Using **Sentinel-1 Synthetic Aperture Radar (SAR)** datasets spanning 95 flood events across 42 countries (**MMFlood Dataset via IEEE DataPort**), the pipeline structures raw polarimetric backscatter measurements into highly reproducible 4D tensor pipelines. It then trains and benchmarks advanced deep semantic segmentation architectures to reliably isolate flooded areas under all-weather, day-and-night conditions.

## Tech Stack & Ecosystem
* **Core Languages & Platforms**: Python 3.10, Google Colab Pro / Jupyter Notebooks
* **Geospatial & Vector Engineering**: GDAL (Geospatial Data Abstraction Library), OGR
* **Data Processing & Computer Vision**: OpenCV (cv2), NumPy, Pandas
* **Deep Learning Frameworks**: TensorFlow 2.x, Keras API (Functional Framework)
* **Model Validation**: Scikit-Learn (sklearn.metrics)

## Data Engineering & Preprocessing Pipeline

Geospatial data manipulation and structural feature formatting are central components of this architecture:

1. **Multi-Channel Vector Vectorization & Rasterization**: Leveraged **GDAL** to intercept geospatial ground truth boundaries and shapefiles, converting disparate coordinates to structured raster grids bound directly to pixel spatial bounds.
2. **Polarimetric Signal Integration**: Isolated the **VV (vertical-transmit, vertical-receive)** and **VH (vertical-transmit, horizontal-receive)** backscatter signals from the wide-swath (IW) sensor arrays, eliminating weather/cloud obstructions.
3. **Sliding-Window Matrix Tiling**: Processed massive satellite matrices by implementing automated sliding-window tile slicing to generate clean, normalized $256 \times 256$ micro-tensors. 
4. **Unified 4D Tensor Pipeline**: Bundled individual bands into structured 4D float tensors shaped `(Batch_Size, Width, Height, Channels)` to facilitate accelerated vectorized loading during model training.

## Neural Network Architectures Evaluated

Four specialized fully convolutional networks (FCNs) were evaluated to assess feature localization accuracy against spatial resolution decay:

* **U-Net**: A traditional encoder-decoder segmentation backbone utilizing multi-level skip connections to concatenate localized structural details directly back into upscale maps.
* **Recurrent Residual U-Net (R2U-Net)**: Replaces standard convolution layers with recurrent residual blocks to cultivate iterative, scale-invariant feature extraction.
* **Attention U-Net**: Introduces gating mechanisms inside the skip pathways to dynamically recalibrate backpropagated gradients, suppressing irrelevant background noise.
* **Attention Residual U-Net (Attention Res U-Net)**: Combines recurrent depth dependencies with attention filtering matrices for highly localized boundary resolution.

## Performance Benchmarks & Validation Analytics

Models were trained for **50 epochs** with a **batch size of 64**, driven by the **Adam optimizer** and monitored using a localized binary cross-entropy loss function. Validation metrics were rigorously evaluated over independent, out-of-sample holdout datasets.

### Model Metrics Comparison
| Model Architecture | Mean Test Accuracy | Mean Precision | Mean Recall | Mean F1-Score | **Mean Jaccard (IoU)** | Training Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard U-Net** | **0.8956** | **0.8410** | **0.9240** | **0.8805** | **0.7722** | ~6 mins |
| **Attention Res U-Net**| 0.7892 | 0.7211 | 0.7949 | 0.7549 | 0.6013 | ~16 mins |
| **Attention U-Net** | 0.7698 | 0.7058 | 0.7909 | 0.7444 | 0.5790 | ~14 mins |
| **R2U-Net** | 0.7235 | 0.6843 | 0.7481 | 0.7131 | 0.5359 | ~13 mins |

### Key Production Insights
* **Data Efficiency**: Standard U-Net demonstrated superior generalizability and structural stability over limited datasets, reaching a baseline mean Jaccard overlap score of **0.7722**.
* **Attention Localizations**: While parametric Attention architectures require a larger dataset to fully mature their weighting layers, they visually produced substantially cleaner segmentation boundaries along intricate water-land transitions, effectively mitigating urban radar noise.

## Scalable Production-Ready Roadmap (Data Engineer Perspectives)
To transition this standalone notebook environment into a live, industrial enterprise pipeline, the following scalable strategies are planned:
1. **Cloud Decoupling & Storage**: Moving batch loading blocks from cloud notebooks over to an optimized cloud storage layout (e.g., AWS S3 or Google Cloud Storage) with partition-based spatial directory configurations.
2. **Containerized Orchestration**: Wrapping core preprocessing engines inside standalone **Docker** profiles and executing task runs via **Apache Airflow** or **Prefect** DAGs.
3. **Data Stream Ingestion**: Replacing manual downloads with scheduled sensor lookups pointing to live **Copernicus Open Access Hub APIs** to generate real-time local updates.
4. **Distributed Inference & Serving**: Moving trained validation binaries behind an asynchronous **FastAPI** web proxy scale or registering artifacts to a **TFX (TensorFlow Extended)** pipeline infrastructure.

