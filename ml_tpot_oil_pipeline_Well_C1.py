import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=123)

# Average CV score on the training set was: -0.1280628168748001
exported_pipeline = make_pipeline(
    Nystroem(gamma=0.05, kernel="linear", n_components=4),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    # MaxAbsScaler(),
    PCA(iterated_power=6, svd_solver="randomized"),
    LassoLarsCV(normalize=False)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 123)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
