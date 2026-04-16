"""Application controller that coordinates loading, analysis, plotting, and export."""

from __future__ import annotations

from opengems.analysis import IsochronalInterpolator, VelocityFieldCalculator
from opengems.export import FigureExporter
from opengems.io import ActivationMatrixLoader
from opengems.models import GridConfig
from opengems.plotting import MatplotlibMapBuilder, PlotlyMapBuilder


class OpenGEMSController:
    """Coordinate the full OpenGEMS workflow."""

    def __init__(self) -> None:
        """Initialize all service objects used by the application."""
        self.loader = ActivationMatrixLoader()
        self.interpolator = IsochronalInterpolator()
        self.velocity_calculator = VelocityFieldCalculator()
        self.plotly_builder = PlotlyMapBuilder()
        self.matplotlib_builder = MatplotlibMapBuilder()
        self.exporter = FigureExporter()

    def run_analysis(self, uploaded_file: object, grid_config: GridConfig, color_scale: str) -> dict:
        """Run the full analysis pipeline.

        Args:
            uploaded_file: Uploaded CSV file object.
            grid_config: User-defined grid configuration.
            color_scale: Selected colormap name.

        Returns:
            A dictionary containing tables, figures, and export bytes.
        """
        raw_matrix = self.loader.load_csv(uploaded_file)
        activation_data = self.loader.fit_to_grid(raw_matrix, grid_config)

        interpolated_field = self.interpolator.interpolate(activation_data, grid_config)
        velocity_field = self.velocity_calculator.compute(interpolated_field)

        iso_fig = self.plotly_builder.build_isochronal_figure(interpolated_field, color_scale)
        vel_fig = self.plotly_builder.build_velocity_figure(velocity_field, color_scale)

        iso_export_figure = self.matplotlib_builder.build_isochronal_export_figure(
            interpolated_field,
            color_scale,
        )
        iso_jpg = self.exporter.to_jpg_bytes(iso_export_figure)

        iso_export_figure = self.matplotlib_builder.build_isochronal_export_figure(
            interpolated_field,
            color_scale,
        )
        iso_svg = self.exporter.to_svg_bytes(iso_export_figure)

        vel_export_figure = self.matplotlib_builder.build_velocity_export_figure(
            velocity_field,
            color_scale,
        )
        vel_jpg = self.exporter.to_jpg_bytes(vel_export_figure)

        vel_export_figure = self.matplotlib_builder.build_velocity_export_figure(
            velocity_field,
            color_scale,
        )
        vel_svg = self.exporter.to_svg_bytes(vel_export_figure)

        return {
            "matrix": activation_data.padded_matrix,
            "iso_fig": iso_fig,
            "vel_fig": vel_fig,
            "iso_jpg": iso_jpg,
            "iso_svg": iso_svg,
            "vel_jpg": vel_jpg,
            "vel_svg": vel_svg,
            "pad_applied": (
                raw_matrix.shape[0] < grid_config.num_h or raw_matrix.shape[1] < grid_config.num_w
            ),
            "crop_applied": (
                raw_matrix.shape[0] > grid_config.num_h or raw_matrix.shape[1] > grid_config.num_w
            ),
        }