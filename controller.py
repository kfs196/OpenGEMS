"""Application controller that coordinates loading, analysis, plotting, and export."""

from __future__ import annotations

from opengems.analysis import IsochronalInterpolator, VelocityFieldCalculator
from opengems.export import FigureExporter
from opengems.io import ActivationMatrixLoader
from opengems.models import GridConfig
from opengems.plotting import MatplotlibMapBuilder, PlotlyMapBuilder
from lib_fevt.fevt import FEVTConfig, FEVTDetector

from typing import Dict, Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    

class FEVTController:
    """Coordinate the full workflow of FEVT Algorithm."""

    def __init__(self) -> None:
        """Initialize all service objects used by the application."""
        pass

  
    def plot_results_plotly(self, result: Dict[str, Any]) -> go.Figure:
        """Build an interactive Plotly figure for signal-processing results.

        The layout keeps the original 4-panel structure:
        1. Input signal with detected activation times
        2. Smoothed detection signal
        3. Falling-edge detector signal
        4. FEVT signal and variable threshold

        Args:
            result: A dictionary containing time-series arrays and detected indices.

        Returns:
            A Plotly Figure object ready for display in Streamlit.
        """
        t = np.asarray(result["time"])
        sig = np.asarray(result["signal"])
        s = np.asarray(result["smoothed_signal"])
        e = np.asarray(result["edge_signal"])
        f = np.asarray(result["fevt_signal"])
        thr = np.asarray(result["threshold"])
        at_idx = np.asarray(result["activation_idx"], dtype=int)

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=(
                "Input signal with detected activation times",
                "Smoothed detection signal S(t)",
                "Falling-edge detector E(t)",
                "FEVT signal F(t) and variable threshold",
            ),
        )

        # Panel 1: raw signal + activation markers
        fig.add_trace(
            go.Scatter(
                x=t,
                y=sig,
                mode="lines",
                name="Input signal",
                line=dict(width=1.5),
                hovertemplate="Time: %{x:.4f}s<br>Signal: %{y:.4f}uV<extra></extra>",
            ),
            row=1,
            col=1,
        )

        if len(at_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=t[at_idx],
                    y=sig[at_idx],
                    mode="markers",
                    name="Activation times",
                    marker=dict(size=7, color="orange", symbol="circle"),
                    hovertemplate="Time: %{x:.4f}s<br>Signal: %{y:.4f}uV<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Panel 2: smoothed signal
        fig.add_trace(
            go.Scatter(
                x=t,
                y=s,
                mode="lines",
                name="Smoothed signal",
                line=dict(width=1.5),
                hovertemplate="Time: %{x:.4f}s<br>S(t): %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Panel 3: edge signal + zero line
        fig.add_trace(
            go.Scatter(
                x=t,
                y=e,
                mode="lines",
                name="Edge signal",
                line=dict(width=1.5),
                hovertemplate="Time: %{x:.4f}s<br>E(t): %{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=np.zeros_like(t),
                mode="lines",
                name="Zero line",
                line=dict(width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # Panel 4: FEVT signal + threshold
        fig.add_trace(
            go.Scatter(
                x=t,
                y=f,
                mode="lines",
                name="FEVT signal",
                line=dict(width=1.5),
                hovertemplate="Time: %{x:.4f}s<br>F(t): %{y:.4f}<extra></extra>",
            ),
            row=4,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=thr,
                mode="lines",
                name="Variable threshold",
                line=dict(width=1.5, dash="dash"),
                hovertemplate="Time: %{x:.4f}s<br>Threshold: %{y:.4f}<extra></extra>",
            ),
            row=4,
            col=1,
        )

        # Axis labels
        fig.update_yaxes(title_text="uV", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=4, col=1)
        fig.update_xaxes(title_text="Time (s)", row=4, col=1)

        # Clean visual style
        fig.update_layout(
            height=950,
            template="plotly_white",
            margin=dict(l=60, r=30, t=80, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
            ),
            hovermode="x unified",
        )

        # Light gridlines for readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.08)")

        return fig



    def run_analysis(self, uploaded_file: object, fevt_config: FEVTConfig)  -> dict:

        """Run the full analysis pipeline.

        Args:
            uploaded_file: Uploaded CSV file object.
            grid_config: User-defined configuration of the FEVT Algorithm.
            color_scale: Selected colormap name.

        Returns:
            A dictionary containing tables, figures, and export bytes.
        """

        cfg = fevt_config
        t, v = FEVTDetector.load_csv_signal(uploaded_file)
        the_fs = FEVTDetector.infer_fs_from_time(t, cfg.fs)

        if cfg.is_prefilter:
            v_proc = FEVTDetector.bandpass_filter(v, fs= the_fs)
        else:
            v_proc = v.copy()
        
        detector = FEVTDetector(config= cfg)
        the_result = detector.detect(v_proc, t)
        go_fig = self.plot_results_plotly(the_result)

        the_result.update({'fig': go_fig})

        return the_result


    

        


        



    # def main() -> None:
    # args = parse_args()
    # os.makedirs(args.output_dir, exist_ok=True)

    # if args.input_csv is None:
    #     t, v = make_demo_signal(fs=args.fs)
    #     input_mode = "demo"
    #     fs = args.fs
    # else:
    #     detector_for_io = FEVTDetector(FEVTConfig(fs=args.fs))
    #     t, v = detector_for_io.load_csv_signal(args.input_csv)
    #     fs = infer_fs_from_time(t, args.fs)
    #     input_mode = "csv"

    # cfg = FEVTConfig(
    #     fs=fs,
    #     transform=args.transform,
    #     smoothing_sec=args.smoothing_sec,
    #     threshold_multiplier=args.threshold_multiplier,
    #     refractory_sec=args.refractory_sec,
    #     running_half_width_sec=args.running_half_width_sec,
    #     min_event_amplitude_uv=args.min_event_amplitude_uv,
    # )
    # detector = FEVTDetector(cfg)

    # if args.prefilter:
    #     v_proc = detector.bandpass_filter(v, fs=fs)
    # else:
    #     v_proc = v.copy()

    # result = detector.detect(v_proc, t)

    # csv_path = os.path.join(args.output_dir, "activation_times.csv")
    # json_path = os.path.join(args.output_dir, "run_metadata.json")
    # png_path = os.path.join(args.output_dir, "fevt_diagnostic.png")

    # save_activation_times_csv(csv_path, result["activation_times"])
    # plot_results(png_path, result)

    # metadata = {
    #     "input_mode": input_mode,
    #     "n_samples": int(len(t)),
    #     "fs": float(fs),
    #     "n_events": int(len(result["activation_idx"])),
    #     "activation_times_sec": [float(x) for x in result["activation_times"]],
    #     "config": json.loads(result["config"][0]),
    # }
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, ensure_ascii=False, indent=2)

    # print(f"Detected {metadata['n_events']} activation times")
    # print(f"CSV:  {csv_path}")
    # print(f"JSON: {json_path}")
    # print(f"PNG:  {png_path}")