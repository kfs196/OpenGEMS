"""Streamlit entry point for OpenGEMS."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from controller import OpenGEMSController, FEVTController
from opengems.models import GridConfig
from lib_fevt.fevt import FEVTDetector
from lib_fevt.fevt import FEVTConfig

fevt_config = FEVTConfig(30) # Init the data class of FEVT algorithm


class OpenGEMSApp:
    """Define and render the Streamlit user interface."""

    def __init__(self) -> None:
        """Create the UI application with a single controller instance."""
        
        self.controller1 = FEVTController()
        self.controller2 = OpenGEMSController()

    def render_header(self) -> None:
        """Render the page title and a short project description."""
        st.title("OpenGEMS :flashlight:")
        st.markdown(
            """
            A lightweight open-source tool for gastrointestinal electrophysiological signal analysis and data visualization on 2D electrode grids
            (Version: 0.2)
            """
        )

    def render_inputs_module1(self) -> tuple[GridConfig | None, object | None, str | None, bool]:
        
        with st.form("fevt_form"):
            st.header("Input Configuration of Module I")
            st.markdown(
                """
                **Current scope I: Gastric slow wave events detection**
                - Import single-channel gastrointestinal electrical time-domain signals
                - Detect slow wave activation times through a series of signal processing steps
                - Visualize and export detection results
                """
            )

            # def parse_args() -> argparse.Namespace:
            #     parser = argparse.ArgumentParser(description="Single-channel FEVT slow-wave activation-time detector")
            #     parser.add_argument("--input_csv", type=str, default=None,
            #                         help="CSV path with two columns: time_sec, signal")
            #     parser.add_argument("--output_dir", type=str, default="outputs")
            #     parser.add_argument("--fs", type=float, default=30.0,
            #                         help="Sampling rate. Used for demo mode or if time column is uniformly sampled.")
            #     parser.add_argument("--transform", type=str, default="NEO", choices=["ND", "ASD", "NEO", "DEO4"])
            #     parser.add_argument("--smoothing_sec", type=float, default=1.0)
            #     parser.add_argument("--threshold_multiplier", type=float, default=4.0)
            #     parser.add_argument("--refractory_sec", type=float, default=7.0)
            #     parser.add_argument("--running_half_width_sec", type=float, default=10.0)
            #     parser.add_argument("--min_event_amplitude_uv", type=float, default=None)
            #     parser.add_argument("--prefilter", action="store_true",
            #                         help="Apply 2nd-order Butterworth bandpass 1 cpm to 60 cpm before FEVT.")
            #     return parser.parse_args()

            uploaded_file = st.file_uploader(
                "Data File of a Time-Series Signal (.csv)",
                type=["csv"],
                help="Acceptable CSV have two columns: time_sec, signal"
            )

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                the_fs = st.number_input(
                    "Sampling Rate (Hz)",
                    min_value=5,
                    value=30,
                    step=5
                )
                the_threshold_multiplier = st.number_input(
                    "Threshold Multiplier (int)",
                    min_value= 1.0,
                    value= 4.0,
                    step= 1.0
                )
                the_edge_kernel_sec = st.number_input(
                    "Duration of Edge Detection (s)",
                    min_value= 0.5,
                    max_value= 2.0,
                    value= 1.0,
                    step= 0.1
                )

            with col_b:
                the_transform = st.selectbox(
                    "Transform Algorithm",
                    options=["ND", "ASD", "NEO", "DEO4"],
                    index=2,
                )
                the_refractory_sec = st.number_input(
                    "Refractory Time (s)",
                    min_value= 1.0,
                    value= 7.0,
                    step= 1.0
                )
                the_is_prefilter = st.selectbox(
                    "Apply 2nd-order Butterworth bandpass",
                    options=[True, False],
                    index= 0
                )

            with col_c:
                the_smoothing_sec = st.number_input(
                    "Smoothing Seconds (s)",
                    min_value=0.1,
                    value=1.0,
                    step=0.05
                )
                the_running_half_width_sec = st.number_input(
                    "Half Width of Threshold (s)",
                    min_value=1.0,
                    value=10.0,
                    step=1.0
                )

            fevt_config = FEVTConfig(
                fs= the_fs,
                transform= the_transform,
                smoothing_sec= the_smoothing_sec,
                threshold_multiplier= the_threshold_multiplier,
                refractory_sec= the_refractory_sec,
                running_half_width_sec= the_running_half_width_sec,
                edge_kernel_sec= the_edge_kernel_sec,
                is_prefilter= the_is_prefilter
            )

            submittedI = st.form_submit_button("Run Analysis")

            if not submittedI:
                return None, None, False
            
            if uploaded_file is None:
                st.warning("Please upload a CSV file before running the analysis.")
                return None, None, True

            return fevt_config, uploaded_file, True


    def render_inputs_module2(self) -> tuple[GridConfig | None, object | None, bool]:
        """Render all input widgets and return the submitted configuration."""

        with st.form("opengems_form"):
            st.header("Input Configuration of Module II")
            st.markdown(
                """
                **Current scope II: 2D electrode grids data mapping**
                - Load a 2D activation-time matrix from CSV
                - Build an interpolated isochronal map
                - Estimate an apparent velocity field from the activation-time gradient
                - Export figures as JPG and SVG
                """
            )

            col_a, col_b = st.columns(2)
            with col_a:
                num_h = st.number_input(
                    "Number of Rows (int)",
                    min_value=1,
                    value=8,
                    step=1,
                )
                distance_h = st.number_input(
                    "Row Spacing (mm)",
                    min_value=0.001,
                    value=5.0,
                    step=0.1,
                    format="%.3f",
                )

            with col_b:
                num_w = st.number_input(
                    "Number of Columns (int)",
                    min_value=1,
                    value=8,
                    step=1,
                )
                distance_w = st.number_input(
                    "Column Spacing (mm)",
                    min_value=0.001,
                    value=5.0,
                    step=0.1,
                    format="%.3f",
                )

            color_scale = st.selectbox(
                "Color Scale",
                options=["rainbow", "coolwarm", "viridis"],
                index=0,
            )

            uploaded_file = st.file_uploader(
                "Activation Time Matrix (.csv)",
                type=["csv"],
            )

            submittedII = st.form_submit_button("Run Analysis")

        if not submittedII:
            return None, None, None, False

        if uploaded_file is None:
            st.warning("Please upload a CSV file before running the analysis.")
            return None, None, None, True

        grid_config = GridConfig(
            num_h=int(num_h),
            num_w=int(num_w),
            distance_h=float(distance_h),
            distance_w=float(distance_w),
        )
        return grid_config, uploaded_file, color_scale, True
    

    def render_results1(self, results: dict) -> None:
        """Render analysis outputs in a clean top-down layout (Module I)."""

        st.divider()
        st.header("Analysis Results of Module I :star:")
        st.subheader("Detection Signal Transform Outcomes")
        act_times = pd.DataFrame(results['activation_times'], columns=['Activation Times (s)'])
        st.dataframe(act_times, width=300)

        st.subheader("Illustration of signal processing")
        plotly_config = {
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }
        st.plotly_chart(
            results['fig'],
            use_container_width=True,
            config=plotly_config,
        )
        pass

    def render_results2(self, results: dict) -> None:
        """Render analysis outputs in a clean top-down layout (Module II)."""
        if results["pad_applied"]:
            st.info("The input matrix was smaller than the target grid and was padded with zeros for display.")
        if results["crop_applied"]:
            st.warning("The input matrix was larger than the target grid and was cropped to fit the selected workspace.")

        st.divider()
        st.header("Analysis Results of Module II :star:")

        st.subheader("Loaded Activation Matrix")
        st.dataframe(pd.DataFrame(results["matrix"]), use_container_width=True)

        plotly_config = {
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "scrollZoom": True,
        }

        st.subheader("Interpolated Isochronal Map")
        st.caption("Use hover to inspect values. Zoom, pan, and reset are available in the toolbar.")
        st.plotly_chart(results["iso_fig"], use_container_width=True, config=plotly_config)

        iso_col1, iso_col2 = st.columns(2)
        with iso_col1:
            st.download_button(
                label="Download Isochronal Map (JPG)",
                data=results["iso_jpg"],
                file_name="isochronal_map.jpg",
                mime="image/jpeg",
            )
        with iso_col2:
            st.download_button(
                label="Download Isochronal Map (SVG)",
                data=results["iso_svg"],
                file_name="isochronal_map.svg",
                mime="image/svg+xml",
            )

        st.subheader("Velocity Field Map")
        st.caption("The background shows speed magnitude; arrows show propagation direction.")
        st.plotly_chart(results["vel_fig"], use_container_width=True, config=plotly_config)

        vel_col1, vel_col2 = st.columns(2)
        with vel_col1:
            st.download_button(
                label="Download Velocity Map (JPG)",
                data=results["vel_jpg"],
                file_name="velocity_field_map.jpg",
                mime="image/jpeg",
            )
        with vel_col2:
            st.download_button(
                label="Download Velocity Map (SVG)",
                data=results["vel_svg"],
                file_name="velocity_field_map.svg",
                mime="image/svg+xml",
            )

    def run(self) -> None:
        """Run the full Streamlit page workflow."""
        st.set_page_config(page_title="OpenGEMS-V0.1", layout="wide")
        self.render_header()
        module_flagI = False
        module_flagII = False

        the_fevt_config, uploaded_file_I, submittedI = self.render_inputs_module1()
        the_grid_config, uploaded_file_II, color_scale, submittedII = self.render_inputs_module2()

        if not submittedI or the_fevt_config is None or uploaded_file_I is None:
            module_flagI = False
        else:
            module_flagI = True
        
        if not submittedII or the_grid_config is None or uploaded_file_II is None or color_scale is None:
            module_flagII = False
        else:
            module_flagII = True
        
        if module_flagI:
            resultsI = self.controller1.run_analysis(
                uploaded_file=uploaded_file_I,
                fevt_config=the_fevt_config
            )
            self.render_results1(resultsI)

        if module_flagII:
            resultsII = self.controller2.run_analysis(
                uploaded_file=uploaded_file_II,
                grid_config=the_grid_config,
                color_scale=color_scale,
            )
            self.render_results2(resultsII)


if __name__ == "__main__":
    app = OpenGEMSApp()
    app.run()