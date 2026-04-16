"""Numerical analysis methods for interpolation and velocity estimation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

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
        base_x, base_y = np.meshgrid(grid_config.x_coords, grid_config.y_coords)

        valid_points = activation_data.valid_mask
        points = np.column_stack((base_x[valid_points], base_y[valid_points]))
        values = activation_data.padded_matrix[valid_points]

        dense_w = max(grid_config.num_w * 20, self.min_dense_points)
        dense_h = max(grid_config.num_h * 20, self.min_dense_points)

        dense_x = np.linspace(grid_config.x_coords.min(), grid_config.x_coords.max(), dense_w)
        dense_y = np.linspace(grid_config.y_coords.min(), grid_config.y_coords.max(), dense_h)
        dense_xx, dense_yy = np.meshgrid(dense_x, dense_y)

        # First use linear interpolation for a smooth surface.
        linear_surface = griddata(points, values, (dense_xx, dense_yy), method="linear")

        # Then fill any edge NaNs with nearest-neighbor interpolation.
        nearest_surface = griddata(points, values, (dense_xx, dense_yy), method="nearest")
        surface = np.where(np.isnan(linear_surface), nearest_surface, linear_surface)

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
        grad_sq[grad_sq < 1e-12] = np.nan

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