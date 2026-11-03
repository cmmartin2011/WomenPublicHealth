README - Statistical Characterization of Women’s Public Health in the Americas During and After COVID-19
==============================================================

This repository contains a Python library for computing disjoint orthogonal components 
in the three loading matrices of the Tucker3 model and for constructing convex hulls 
based on multivariate data. Additionally, this Python code can be used to generate the biplots
and implement the three-way k-means clustering analysis.

--------------------------------------------------------------
Usage Instructions
--------------------------------------------------------------

1. Load the data:
   _, _, datamatrix = get_datamatrix_csv2(r"DataTensorPaperWomen.csv", center=False, scale=False)

2. Construct the data tensor:
   X = get_tensor_row(datamatrix, 10, 5, 4)

3. Perform centering and scaling in mode B (variable mode):
   centerB(X)
   scalingB(X)

4. Build the Tucker3 models used to elaborate the convex hull:
   Models = TableFits(X, 5, 4, 3, 150)

5. Generate the convex hull:
   plot_ch(Models)

6. Construct the three disjoint loading matrices A, B, and C:
   A, B, C, GA, pFit, execution_time = disjoint_T3_ABC(X, 3, 3, 2, 150, 0.00001)

7. To generate the biplots, run the file DynamicBiplots.py:
   Python DynamicBiplots.py

8. For the three-way k-means clustering analysis, run the file ThreeWay-KMeans.py
   Python ThreeWay-KMeans.py

--------------------------------------------------------------
Notes
--------------------------------------------------------------
- Ensure that the datasets "DataTensorPaperWomen.csv" and "DataTensorPaperWomen.xlsx" are
  located in the working directory.
- It is recommended to run the function disjoint_T3_ABC between 5 and 10 times and take the best solution.
- Execution time may vary depending on dataset size and system performance.
