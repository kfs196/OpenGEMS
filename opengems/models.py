"""Core data models used by OpenGEMS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridConfig:
    """Store the electrode grid shape and physical spacing.

    Attributes:
        num_h: Number of rows in the 2D electrode grid.
        num_w: Number of columns in the 2D electrode grid.
        distance_h: Physical spacing between adjacent rows.
        distance_w: Physical spacing between adjacent columns.
    """

    num_h: int
    num_w: int
    distance_h: float
    distance_w: float

    @property
    def x_coords(self) -> np.ndarray:
        """Return the physical x-coordinates of all columns."""
        return np.arange(self.num_w, dtype=float) * self.distance_w

    @property
    def y_coords(self) -> np.ndarray:
        """Return the physical y-coordinates of all rows."""
        return np.arange(self.num_h, dtype=float) * self.distance_h


@dataclass
class ActivationData:
    """Store the loaded activation-time matrix and its valid-data mask.

    Attributes:
        raw_matrix: Original matrix read from the CSV file.
        padded_matrix: Matrix resized to the target grid shape.
        valid_mask: Boolean mask marking entries that came from the original data.
    """

    raw_matrix: np.ndarray
    padded_matrix: np.ndarray
    valid_mask: np.ndarray


@dataclass
class InterpolatedField:
    """Store a dense interpolated activation-time surface.

    Attributes:
        x_coords: Dense x-axis coordinates.
        y_coords: Dense y-axis coordinates.
        values: Interpolated 2D activation-time field.
    """

    x_coords: np.ndarray
    y_coords: np.ndarray
    values: np.ndarray


@dataclass
class VelocityField:
    """Store the apparent velocity field derived from activation times.

    Attributes:
        x_coords: Dense x-axis coordinates.
        y_coords: Dense y-axis coordinates.
        vx: Velocity x-component.
        vy: Velocity y-component.
        speed: Velocity magnitude.
    """

    x_coords: np.ndarray
    y_coords: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    speed: np.ndarray