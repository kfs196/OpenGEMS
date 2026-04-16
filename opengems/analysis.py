"""Numerical analysis methods for interpolation and velocity estimation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from opengems.models import ActivationData, GridConfig, InterpolatedField, VelocityField


class IsochronalInterpolator:
    """Create a dense interpolated activation-time surface from grid data."""

    def __init__(self, min_dense_points: int = 200) -> None:
        """Initialize the interpolator.

        Args:
            min_dense_points: Minimum number of dense sampling points per axis.
        """
        self.min_dense_points = min_dense_points

    def interpolate(self, activation_data: ActivationData, grid_config: GridConfig) -> InterpolatedField:
        """Interpolate the valid activation-time samples onto a dense grid.

        Args:
            activation_data: Loaded matrix and valid-data mask.
            grid_config: Electrode grid configuration.

        Returns:
            A dense interpolated activation-time field.
        """

        valid_mask = activation_data.valid_mask
        valid_rows = np.where(valid_mask.any(axis=1))[0] # Find the valid rectangular region.
        valid_cols = np.where(valid_mask.any(axis=0))[0]
        if len(valid_rows) == 0 or len(valid_cols) == 0:
            raise ValueError("No valid activation data are available for interpolation.")
        
        row_start, row_end = valid_rows[0], valid_rows[-1] + 1
        col_start, col_end = valid_cols[0], valid_cols[-1] + 1
        sub_values = activation_data.padded_matrix[row_start:row_end, col_start:col_end]
        sub_y = grid_config.y_coords[row_start:row_end]
        sub_x = grid_config.x_coords[col_start:col_end]

        kx = min(3, len(sub_x) - 1) # Use cubic spline when enough samples are available.
        ky = min(3, len(sub_y) - 1) # Fall back to quadratic or linear when the valid region is small.

        dense_w = max(len(sub_x) * 20, self.min_dense_points)
        dense_h = max(len(sub_y) * 20, self.min_dense_points)
        dense_x = np.linspace(sub_x.min(), sub_x.max(), dense_w)
        dense_y = np.linspace(sub_y.min(), sub_y.max(), dense_h)

        spline = RectBivariateSpline(sub_y, sub_x, sub_values, ky=ky, kx=kx)
        surface = spline(dense_y, dense_x)

        return InterpolatedField(
            x_coords=dense_x,
            y_coords=dense_y,
            values=surface,
        )


class VelocityFieldCalculator:
    """Estimate an apparent propagation velocity field from activation time."""

    def compute(self, interpolated_field: InterpolatedField) -> VelocityField:
        """Compute velocity components and speed magnitude.

        The calculation uses the spatial gradient of the activation-time field.
        The apparent propagation direction follows the negative time gradient.

        Args:
            interpolated_field: Dense activation-time surface.

        Returns:
            A velocity field with vx, vy, and speed.
        """
        dT_dy, dT_dx = np.gradient(
            interpolated_field.values,
            interpolated_field.y_coords,
            interpolated_field.x_coords,
        )

        grad_sq = dT_dx**2 + dT_dy**2
        threshold = np.percentile(grad_sq[~np.isnan(grad_sq)], 0.5) # Calculate the 0.5th percentile in the array as the threshold 
        grad_sq[grad_sq < threshold] = np.nan

        vx = dT_dx / grad_sq
        vy = dT_dy / grad_sq
        speed = np.sqrt(vx**2 + vy**2)

        return VelocityField(
            x_coords=interpolated_field.x_coords,
            y_coords=interpolated_field.y_coords,
            vx=vx,
            vy=vy,
            speed=speed,
        )