# **This README describes the workflow for `Meron_ML Taxon Classifier_GSoC 2026_Prop.ipynb`.**

# ML Taxon Classifier Proposal – GSoC 2026

This file describes my step-by-step plan for building a machine-learning classifier to assign malaria mosquito samples to major taxa using raw sequencing reads.

---

## 1. Filter Samples

I will start by using the MalariaGEN AG3 metadata to filter samples from **East African countries**.  
This will give me the subset of samples I want to analyze, focusing on countries like Ethiopia, Kenya, Tanzania, Uganda, etc.

---

## 2. Access ENA FASTQ Links

Using the `sample_id` column of the filtered metadata, I will retrieve **ENA run accessions** and generate links to the raw FASTQ files.  
I will not download these files locally but instead use **Colab** to access them directly via HTTP/FTP links.

---

## 3. Fetch FASTQ Files in Colab

In Colab, I will use `!wget` to fetch FASTQ files directly into the Colab environment without storing them on my local machine.  
This ensures that the analysis can be done even in resource-limited environments.

---

## 4. Filter FASTQ Reads

I will filter the raw FASTQ reads to **retain only high-quality reads**.  
This step ensures that downstream k-mer extraction and ML models are trained on reliable data.

---

## 5. Extract K-mer Features

From the filtered FASTQ reads, I will extract **k-mer counts** for each sample.  
These k-mer features will form the **input matrix** for ML classification.

---

## 6. Split Dataset

I will split the k-mer feature matrix into **training and testing datasets**.  
The `taxon` column in the metadata will serve as the **target label**.

---

## 7. Train Machine Learning Classifiers

I will train several ML models on the training dataset, including:

- Random Forest  
- Logistic Regression  
- Support Vector Machine  
- XGBoost  
- LightGBM  
- K-Nearest Neighbors  
- Decision Tree  
- Gradient Boosting  

I will evaluate the models on the testing set and choose the best-performing one for further work.

---

## 8. Demonstration Notebook

Alongside this proposal, I have included a **Colab-ready notebook** (`Meron_ML Taxon Classifier_GSoC 2026_Prop.ipynb`) that:

- Filters East African samples  
- Shows placeholder ENA FASTQ links  
- Demonstrates k-mer feature extraction on toy data  
- Splits dataset and trains toy classifiers  

This notebook is intended to **demonstrate my workflow and coding approach safely**, without needing to download large FASTQ files.
