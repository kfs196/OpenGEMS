"""Interactive and static plotting utilities for OpenGEMS."""

from __future__ import annotations

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

from opengems.models import InterpolatedField, VelocityField


def build_plotly_colorscale(name: str, steps: int = 256) -> list[list[float | str]]:
    """Convert a Matplotlib colormap into a Plotly colorscale.

    Args:
        name: Matplotlib colormap name.
        steps: Number of color samples.

    Returns:
        A Plotly-compatible colorscale.
    """
    cmap = cm.get_cmap(name, steps)
    colorscale = []

    for idx in range(cmap.N):
        r, g, b, _ = cmap(idx)
        colorscale.append(
            [
                idx / max(cmap.N - 1, 1),
                f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})",
            ]
        )
    return colorscale


class PlotlyMapBuilder:
    """Build interactive Plotly figures for Streamlit display."""

    def build_isochronal_figure(self, field: InterpolatedField, color_scale: str) -> go.Figure:
        """Build an interactive interpolated isochronal map.

        Args:
            field: Dense activation-time field.
            color_scale: Selected colormap name.

        Returns:
            A Plotly figure.
        """
        figure = go.Figure()

        figure.add_trace(
            go.Contour(
                x=field.x_coords,
                y=field.y_coords,
                z=field.values,
                colorscale=build_plotly_colorscale(color_scale),
                contours=dict(
                    coloring="heatmap",
                    showlines=True,
                ),
                colorbar=dict(title="Activation Time (s)"),
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>time: %{z:.4f}<extra></extra>",
                line=dict(width=0.8),
                ncontours=18,
            )
        )

        figure.update_layout(
            title="Interpolated Isochronal Map",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            template="plotly_white",
            dragmode="pan",
            clickmode="event+select",
            margin=dict(l=30, r=30, t=60, b=30),
            yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
        )
        return figure

    def build_velocity_figure(
        self,
        velocity_field: VelocityField,
        color_scale: str,
        arrow_step: int = 14,
    ) -> go.Figure:
        """Build an interactive velocity field map with arrows.

        Args:
            velocity_field: Velocity magnitude and direction data.
            color_scale: Selected colormap name.
            arrow_step: Sampling interval for the arrow grid.

        Returns:
            A Plotly figure.
        """
        figure = go.Figure()

        figure.add_trace(
            go.Heatmap(
                x=velocity_field.x_coords,
                y=velocity_field.y_coords,
                z=velocity_field.speed,
                colorscale=build_plotly_colorscale(color_scale),
                colorbar=dict(title="Speed (mm/s)"),
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>speed: %{z:.4f}<extra></extra>",
            )
        )

        norm = np.sqrt(velocity_field.vx**2 + velocity_field.vy**2)
        unit_vx = velocity_field.vx / norm
        unit_vy = velocity_field.vy / norm

        sample_x = velocity_field.x_coords[::arrow_step]
        sample_y = velocity_field.y_coords[::arrow_step]
        sample_xx, sample_yy = np.meshgrid(sample_x, sample_y)

        sample_u = unit_vx[::arrow_step, ::arrow_step]
        sample_v = unit_vy[::arrow_step, ::arrow_step]
        valid = (~np.isnan(sample_u)) & (~np.isnan(sample_v))

        quiver = ff.create_quiver(
            x=sample_xx[valid],
            y=sample_yy[valid],
            u=sample_u[valid],
            v=sample_v[valid],
            scale=0.18,
            arrow_scale=0.3,
            line_width=1.1,
            name="Direction",
        )

        for trace in quiver.data:
            trace.update(hoverinfo="skip", showlegend=False, line=dict(color="black"))
            figure.add_trace(trace)

        figure.update_layout(
            title="Velocity Magnitude and Direction",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            template="plotly_white",
            dragmode="pan",
            clickmode="event+select",
            margin=dict(l=30, r=30, t=60, b=30),
            yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
        )
        return figure


class MatplotlibMapBuilder:
    """Build static Matplotlib figures for file export."""

    def build_isochronal_export_figure(self, field: InterpolatedField, color_scale: str):
        """Build a static isochronal map for JPG/SVG export.

        Args:
            field: Dense activation-time field.
            color_scale: Selected colormap name.

        Returns:
            A Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(7, 5.5))

        xx, yy = np.meshgrid(field.x_coords, field.y_coords)
        masked_values = np.ma.masked_invalid(field.values)

        contour = ax.contourf(xx, yy, masked_values, levels=18, cmap=color_scale)
        ax.contour(xx, yy, masked_values, levels=18, colors="black", linewidths=0.5, alpha=0.6)

        colorbar = fig.colorbar(contour, ax=ax)
        colorbar.set_label("Activation Time (s)")

        ax.set_title("Interpolated Isochronal Map")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect("equal")
        ax.invert_yaxis()

        fig.tight_layout()
        return fig

    def build_velocity_export_figure(
        self,
        velocity_field: VelocityField,
        color_scale: str,
        arrow_step: int = 14,
    ):
        """Build a static velocity field map for JPG/SVG export.

        Args:
            velocity_field: Velocity magnitude and direction data.
            color_scale: Selected colormap name.
            arrow_step: Sampling interval for the arrow grid.

        Returns:
            A Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(7, 5.5))

        xx, yy = np.meshgrid(velocity_field.x_coords, velocity_field.y_coords)
        masked_speed = np.ma.masked_invalid(velocity_field.speed)

        heatmap = ax.pcolormesh(xx, yy, masked_speed, shading="auto", cmap=color_scale)
        colorbar = fig.colorbar(heatmap, ax=ax)
        colorbar.set_label("Speed (mm/s)")

        norm = np.sqrt(velocity_field.vx**2 + velocity_field.vy**2)
        unit_vx = velocity_field.vx / norm
        unit_vy = velocity_field.vy / norm

        ax.quiver(
            xx[::arrow_step, ::arrow_step],
            yy[::arrow_step, ::arrow_step],
            unit_vx[::arrow_step, ::arrow_step],
            unit_vy[::arrow_step, ::arrow_step],
            color="black",
            pivot="middle",
            scale=22,
            width=0.003,
        )

        ax.set_title("Velocity Magnitude and Direction")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect("equal")
        ax.invert_yaxis()

        fig.tight_layout()
        return fig