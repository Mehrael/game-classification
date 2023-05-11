import numpy as np
from sklearn.preprocessing import LabelEncoder


class CustomLabelEncoder(LabelEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_ = np.array([])

    def fit(self, y):
        super().fit(y)
        self.classes_ = np.append(self.classes_, '<unseen>')
        return self

    def transform(self, y):
        y_transformed = np.array([x if x in self.classes_ else '<unseen>' for x in y])
        return super().transform(y_transformed)

    def fit_transform(self, y):
        return self.fit(y).transform(y)
