"""Streamlit entry point for OpenGEMS."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from opengems.controller import OpenGEMSController
from opengems.models import GridConfig


class OpenGEMSApp:
    """Define and render the Streamlit user interface."""

    def __init__(self) -> None:
        """Create the UI application with a single controller instance."""
        self.controller = OpenGEMSController()

    def render_header(self) -> None:
        """Render the page title and a short project description."""
        st.title("OpenGEMS")
        st.markdown(
            """
            A lightweight open-source tool for isochronal mapping and velocity field visualization
            on 2D electrode grids. (Version:0.1)

            **Current scope**
            - Load a 2D activation-time matrix from CSV
            - Build an interpolated isochronal map
            - Estimate an apparent velocity field from the activation-time gradient
            - Export figures as JPG and SVG
            """
        )

    def render_inputs(self) -> tuple[GridConfig | None, object | None, str | None, bool]:
        """Render all input widgets and return the submitted configuration."""
        with st.form("opengems_form"):
            st.subheader("Input Configuration")

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

            submitted = st.form_submit_button("Run Analysis")

        if not submitted:
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

    def render_results(self, results: dict) -> None:
        """Render analysis outputs in a clean top-down layout."""
        if results["pad_applied"]:
            st.info("The input matrix was smaller than the target grid and was padded with zeros for display.")
        if results["crop_applied"]:
            st.warning("The input matrix was larger than the target grid and was cropped to fit the selected workspace.")

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

        grid_config, uploaded_file, color_scale, submitted = self.render_inputs()
        if not submitted or grid_config is None or uploaded_file is None or color_scale is None:
            return

        results = self.controller.run_analysis(
            uploaded_file=uploaded_file,
            grid_config=grid_config,
            color_scale=color_scale,
        )
        self.render_results(results)


if __name__ == "__main__":
    app = OpenGEMSApp()
    app.run()