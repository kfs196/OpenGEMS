"""Input loading and grid normalization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opengems.models import ActivationData, GridConfig


class ActivationMatrixLoader:
    """Load a CSV activation-time matrix and fit it to a target grid."""

    def load_csv(self, uploaded_file: object) -> np.ndarray:
        """Load a CSV file into a numeric NumPy array.

        Args:
            uploaded_file: A Streamlit uploaded file object.

        Returns:
            A 2D NumPy array of activation times.
        """
        dataframe = pd.read_csv(uploaded_file, header=None)
        return dataframe.to_numpy(dtype=float)

    def fit_to_grid(self, matrix: np.ndarray, grid_config: GridConfig) -> ActivationData:
        """Fit an input matrix to the target grid size.

        The display matrix is always resized to the user-defined grid size.
        A valid-data mask is kept so padded zeros do not participate in interpolation
        or gradient-based velocity estimation.

        Args:
            matrix: Input activation-time matrix.
            grid_config: Target grid configuration.

        Returns:
            An ActivationData object containing the original matrix, the resized matrix,
            and a valid-data mask.
        """
        target_h = grid_config.num_h
        target_w = grid_config.num_w

        fitted = np.zeros((target_h, target_w), dtype=float)
        valid_mask = np.zeros((target_h, target_w), dtype=bool)

        use_h = min(target_h, matrix.shape[0])
        use_w = min(target_w, matrix.shape[1])

        fitted[:use_h, :use_w] = matrix[:use_h, :use_w]
        valid_mask[:use_h, :use_w] = True

        return ActivationData(
            raw_matrix=matrix,
            padded_matrix=fitted,
            valid_mask=valid_mask,
        )