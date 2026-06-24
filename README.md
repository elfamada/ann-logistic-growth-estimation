# ANN Architecture for Logistic Growth Model Parameter Estimation
Integrating Mathematical Modelling and AI for parameter estimation.

This project was developed during an exclusive international training program (Compact Course) between **Universitas Indonesia (UI)** and **Universiti Malaya (UM)**, funded by the **UI EQUITY Grant** and supported by the **IMPACT Lab**. 

## 📌 Project Overview
Traditional mathematical biology relies on deterministic calculus to solve differential equations and estimate parameters. This project bridges traditional mathematical interpretation with data-driven learning by designing an **Artificial Neural Network (ANN)** architecture to accurately estimate critical parameters ($r$ and $K$) of a **Logistic Growth Model**. This hybrid approach is highly applicable to analyzing system behaviors in public health, epidemiology, and environmental contexts.

### 📊 Task Specifications & Experimental Design
Based on the project's official assignment framework, the workflow was strictly executed under the following criteria:

1. **Data Preparation & Simulation:**
   * Generated $N = 1000$ logistic curves across a time horizon of $t = 0, 1, \dots, 30$ (resulting in a 31-dimensional input vector for each sample).
   * Fixed initial population $P_0 = 20$.
   * Randomized parameters via uniform distribution: $r \in [0.05, 0.50]$ and $K \in [500, 2000]$.
   * Dataset split: **70% Training**, **15% Validation**, and **15% Testing**.
   * Feature scaling: Normalized both features ($X$) and target values ($r$ and $K$).

2. **ANN Exact Architecture:**
   * Type: Feed-forward Deep Neural Network.
   * Layer Topology: $31 \rightarrow 64 \rightarrow 64 \rightarrow 32 \rightarrow 2$ (Input $\rightarrow$ Hidden 1 $\rightarrow$ Hidden 2 $\rightarrow$ Hidden 3 $\rightarrow$ Output).
   * Activations: `ReLU` for hidden layers; `Linear` or `Sigmoid` for normalized targets.
   * Optimizer & Hyperparameters: Trained using **Adam Optimizer** with a learning rate of $0.001$, batch size of $32$, over $200$ epochs.

3. **Loss Function Definition:**
   The network optimizes parameters using a customized batch Mean Squared Error (MSE) formulated as:

$$L_{batch} = \frac{1}{B} \sum_{j=1}^{B} \left[ (\hat{r}_j - r_j)^2 + (\hat{K}_j - K_j)^2 \right]$$

4. **Validation & Final Evaluation:**
   * Validation loss curves were strictly monitored without backpropagation to select the best epoch model.
   * Evaluated final test sets using MAE and RMSE metrics for both $r$ and $K$.

## 🧬 Mathematical Framework
The classic Logistic Growth Model is governed by the following first-order non-linear differential equation:

$$\frac{dP}{dt} = rP \left(1 - \frac{P}{K}\right)$$

Where:
*   $P(t)$ = Population size at time $t$
*   $r$ = Intrinsic growth rate (speed of growth)
*   $K$ = Carrying capacity (maximum sustainable population)
*   $P_0$ = Initial population size at $t = 0$

The analytical solution used to generate our system baseline curves is:

$$P(t) = \frac{K}{1 + \left(\frac{K - P_0}{P_0}\right)e^{-rt}}$$

## 🤖 Machine Learning Pipeline & Architecture
Instead of using traditional regression (like Least Squares), an **Artificial Neural Network (ANN)** was trained to map an entire observed growth curve directly to its underlying parameters $[r, K]$.

1. **Data Generation:** Generated 1,000 synthetic logistic growth curves with randomized uniform distributions of $r \in [0.05, 0.5]$ and $K \in [500, 2000]$, given a fixed $P_0 = 20$.
2. **Preprocessing:** Feature scaling implemented using `MinMaxScaler` to normalize the time-series arrays before feeding them into the network.
3. **ANN Architecture:** Built using TensorFlow/Keras, utilizing a deep feedforward structure with `Dense` layers, `ReLU` activation functions, and an output layer designed for multi-output regression (predicting both $r$ and $K$ simultaneously).

## 📊 Results & Visualizations
The model successfully converges, achieving low Mean Squared Error (MSE) on test splits. The trained network can instantaneously generalize and output exact parameters for completely unseen growth data.

Curve Comparison using Sample (Test Data) :
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/d7d6a109-c196-471b-b265-d697722a6c19" />

Loss Curve : 
<img width="787" height="490" alt="image" src="https://github.com/user-attachments/assets/2dea20f3-0a56-419b-96c6-6fcf105bcf21" />

## 🔬 Advanced Mathematical & Sensitivity Analysis

To rigorously evaluate the mathematical boundaries and robustness of the trained model, we conducted four core analytical evaluations: Max Growth Rate, Local Sensitivity, Global Sensitivity, and Extrapolation Analysis.

### 1. Maximum Growth Rate Analysis
In a Logistic Growth Model, the maximum growth rate does not occur at the beginning, but rather at the inflection point of the sigmoidal curve. Mathematically, this point is reached when the population size is exactly half of the carrying capacity:

$$P(t) = \frac{K}{2}$$

Substituting this back into the governing differential equation yields the maximum growth rate ($\left.\frac{dP}{dt}\right|_{max}$):

$$\left.\frac{dP}{dt}\right|_{max} = r \left(\frac{K}{2}\right) \left(1 - \frac{K/2}{K}\right) = \frac{rK}{4}$$

* **Result:**

  <img width="557" height="65" alt="image" src="https://github.com/user-attachments/assets/89eac5af-ba7d-475f-ae9e-da508e569d11" />

* **Analytical Insight:** The ANN model accurately estimated the maximum growth rate at 48.02, demonstrating performance with a minimal absolute error of 1.98.
  
### 2. Local Sensitivity Analysis (LSA)
Local sensitivity measures how a small perturbation in a specific parameter ($r$ or $K$) independently affects the population output $P(t)$ at a given time, while keeping the other parameter constant. This is evaluated using partial derivatives:

$$\text{Sensitivity to } r: \frac{\partial P(t)}{\partial r} = \frac{K \cdot t \cdot \left(\frac{K - P_0}{P_0}\right) e^{-rt}}{\left[1 + \left(\frac{K - P_0}{P_0}\right)e^{-rt}\right]^2}$$

$$\text{Sensitivity to } K: \frac{\partial P(t)}{\partial K} = \frac{1 - \left(\frac{K - P_0}{P_0}\right) e^{-rt} \cdot \left(\frac{2P_0 - K}{P_0}\right)}{\left[1 + \left(\frac{K - P_0}{P_0}\right)e^{-rt}\right]^2} \quad \text{(Derived at specific localized points)}$$

* **Result:**
  <img width="1176" height="581" alt="image" src="https://github.com/user-attachments/assets/f80ffc59-a945-4100-a55b-f05d2d4c1cd6" />

* **Analytical Insight:** LSA revealed that the model is extremely sensitive to changes in $r$ during the mid-exponential phase, whereas the sensitivity to $K$ dominates asymptotically as $t \to 30$ (the plateau phase). The ANN implicitly learned this time-dependent sensitivity, prioritizing different sections of the 31-dimensional input vector during different growth phases.

### 3. Global Sensitivity Analysis (GSA)
Unlike local sensitivity, Global Sensitivity explores the entire parameter space simultaneously, assessing interactions between $r$ and $K$ across their full uniform distribution ranges ($r \in [0.05, 0.50]$, $K \in [500, 2000]$).

* **Result:**
<img width="1182" height="581" alt="image" src="https://github.com/user-attachments/assets/b123f3c0-8625-494d-abf3-8eb47406ee61" />

* **Analytical Insight:** The GSA confirmed that the variance in the reconstructed population curve is jointly driven by both parameters, but their interaction effect is highly non-linear. The ANN proved robust against these non-linear parameter interactions, demonstrating stable multi-output regression without getting biased toward one dominant parameter.

### 4. Extrapolation Analysis
We tested the network's generalization boundaries by feeding it curves generated from parameters completely *outside* the training bounds (e.g., $r > 0.50$ or $K > 2000$), as well as testing its predictive capacity beyond the calibrated time horizon ($t > 30$).

* **Result:**
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/cb868737-bf9c-4add-be5a-1f2dc2bf9189" />

* **Analytical Insight:** The extrapolation tests highlight the boundary limits of a pure data-driven approach. While the ANN exhibits exceptional interpolation accuracy within the trained envelope, it faces standard geometric constraints when extrapolating far outside the distribution limits. This underscores the necessity of a *hybrid approach*—using mathematical bounds to constrain and guide neural network search spaces in real-world public health applications.

## 📈 Model Performance & Evaluation

The trained Deep Forward ANN model was evaluated on the unseen test dataset (15% split) using both **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** metrics. The errors were calculated independently for the growth rate ($r$), carrying capacity ($K$), and the reconstructed population curve ($P(t)$).

The quantitative evaluation results are detailed below:

| Model | Metric | Intrinsic Growth Rate ($r$) | Carrying Capacity ($K$) | Reconstructed Population ($P$) |
| :--- | :--- | :---: | :---: | :---: |
| **Model ANN** | **RMSE** | 0.002892 | 138.809114 | 3.326933 |
| | **MAE** | 0.002266 | 59.954265 | 0.588755 |

### 🔍 Key Insights from Evaluation:
* **High Precision in Growth Rate ($r$):** The model predicts the intrinsic growth rate with an exceptionally low error (MAE of `0.002266`), showing that the network is highly sensitive to the slope and acceleration of the logistic curve.
* **Robust Population Reconstruction ($P$):** When the predicted parameters $(\hat{r}, \hat{K})$ are plugged back into the analytical solution, the reconstructed growth curve deviates from the actual curve by an average of **only 0.588** population units (MAE), proving the model's reliability in public health and environmental forecasting contexts.

## 👥 Group Members (Group 6)

This international collaborative project was conducted by:
* **Elfa Nusuki Amada** (Universitas Indonesia) 
* **Hanny Awlia** (Universitas Indonesia)
* **Anymore Majere** (Universitas Indonesia)
* **Melissa Law Xue Ying** (Universiti Malaya)
* **Nurin Firzana Binti Mohamad Najib** (Universiti Malaya)

Under the funding of the **UI EQUITY Grant** and supervision of the **IMPACT Lab** team.
