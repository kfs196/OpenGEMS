"""Figure export helpers."""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt


class FigureExporter:
    """Export Matplotlib figures to image bytes."""

    def to_jpg_bytes(self, figure) -> bytes:
        """Convert a Matplotlib figure to JPG bytes.

        Args:
            figure: A Matplotlib figure.

        Returns:
            JPG bytes.
        """
        buffer = BytesIO()
        figure.savefig(buffer, format="jpg", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        plt.close(figure)
        return image_bytes

    def to_svg_bytes(self, figure) -> bytes:
        """Convert a Matplotlib figure to SVG bytes.

        Args:
            figure: A Matplotlib figure.

        Returns:
            SVG bytes.
        """
        buffer = BytesIO()
        figure.savefig(buffer, format="svg", bbox_inches="tight")
        buffer.seek(0)
        image_bytes = buffer.getvalue()
        plt.close(figure)
        return image_bytes