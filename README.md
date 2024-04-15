# Thesis work

Light curve classifier using XGBoost. The data used is from "Alert Classification for the ALeRCE Broker System: The Light Curve Classifier", by Sánchez-Sáez, P.; Reyes, I.; Valenzuela, C.; Förster, F.; Eyheramendy, S.; Elorrieta, F.; Bauer, FE.; Cabrera-Vives, G.;  Estévez, PA.; Catelan, M. and others. (Data available https://doi.org/10.5281/zenodo.4279623)

Full thesis report at https://drive.google.com/file/d/1MMfQae3Ykoc3k3yEsKyXtiRHSm46ymPV/view?usp=sharing

The performance of an XGBoost model was studied to analyze its capacity to handle imbalanced data. Parameter optimization and cost-sensitive learning was used.

A modification to the XGBoost training algorithm was done to include balanced bootstrapping, in order to analyze its effect on imbalanced classification. This modification was made specifically on the train.py module of the python-package.
A parameter was added to the train method to include, when said parameter is True, a proccess of balanced bootstrapping to the algorithm. The code for this modification can be found in the modification.py file.
(Note: The method crashes if using this parameter set to True while using device set as cuda)

A Balanced Random Forest model was used for benchmarking since it is the model described by ALeRCE in the aforementioned paper.
