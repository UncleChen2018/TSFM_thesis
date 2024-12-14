# On the Use of Time Series Foundation Models for Error-Bounded Lossy Compression

## Author
**Ji Chen**  
LinkedIn: [Ji Chen](https://www.linkedin.com/in/ji-chen-91862328b/)

Submitted to the Khoury College of Computer Sciences on **December 03, 2024** in partial fulfillment of the requirements for the degree of **M.Sc. in Computer Sciences**.

---

## Abstract
The rapid increase in time series data volumes demands innovative compression techniques that maintain high fidelity and achieve superior compression ratios. This study introduces a novel framework for integrating Time Series Foundation Models (TSFMs) within error-bounded prediction-based lossy compression.

### Key Contributions
1. **Universal Framework for TSFMs in Compression**
   - Developed a **sliding context prediction mechanism** to adapt TSFMs as compression predictors.
   - Ensures consistent prediction accuracy and strict reproducibility during decompression.

2. **Optimized Multi-Scale Quantization**
   - Designed a **multi-scale quantization and lossless compression strategy** tailored to TSFM prediction errors.
   - Achieves significant reductions in compressed file sizes.

3. **Implementation and Evaluation**
   - Implemented the framework using **Chronos**.
   - Conducted extensive testing on the **Large-scale Open Time Series Archive (LOTSA)**.
   - Compared against the state-of-the-art SZ compressor by Liang et al.

### Results
- Achieved **24.1%-78.8% higher compression ratios** for error bounds below 10⁻³.
- Delivered consistent performance across diverse datasets using fixed parameters.
- Demonstrated both the **effectiveness** and **robustness** of the proposed approach.

### Challenges
- Processing time and computing costs for TSFMs currently restrict practical applications.
- Future improvements in TSFM efficiency could make this approach increasingly viable.

---



## Supervisor
**Mario A. Nascimento**  
LinkedIn: [Mario A. Nascimento](https://www.linkedin.com/in/mario-nascimento-a9362a3/)  
**Title:** Professor of the Practice

---
## Main design
See [design & results](https://github.com/UncleChen2018/TSFM_thesis/tree/main/figs).

---

## Project Workflow & File Structure

### Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tsfm-compression.git
   ```
2. Install **Chronos**:
   ```bash
   pip install git+https://github.com/amazon-science/chronos-forecasting
   ```
3. Download and install **SZ3** binary:
   - SZ3 can be downloaded and compiled from its official repository: [SZ3 on GitHub](https://github.com/szcompressor/SZ3).

### Workflow Overview
To replicate the experiments, use the Jupyter Notebooks starting from **3-1 All data Experiment Script.ipynb** and follow the sequence:

1. **3-1 All data Experiment Script.ipynb**:  
   Conduct experiments for all data. Results are stored in `3-1_all_data_results.csv`.

2. **3-2 All para Experiment Script.ipynb**:  
   Run parameter tuning experiments. Results are saved in `3-2_para_results.csv`.

3. **3-3 All model Experiment Script.ipynb**:  
   Run all model experiments. Results are saved in `3-3_all_model_results.csv`.

4. **3-4 Largest model Experiment.ipynb**:  
   Test the largest model experiment. Results are stored in `3-4_large_model_results.csv`.


   
