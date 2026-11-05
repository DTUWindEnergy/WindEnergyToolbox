from abc import ABC, abstractmethod
from pathlib import Path
import re

from wetb.dlb.report_generator.dlb_utils import normalize_dlc_filter


class ReportSection(ABC):
    """Base class for one report section generated from a report."""

    key = None
    # Dataset names this section may need when comparing two report cases.
    dataset_keys = ()
    # LaTeX assembly details belong to the section that produces the files.
    latex_title = None
    latex_table_subdirectory = None
    latex_figure_subdirectory = None

    def __init__(self, report):
        self.report = report

    @abstractmethod
    def is_available(self):
        """Return whether this section has all required input data."""

    @abstractmethod
    def generate(self, component_name, component):
        """Generate this section's tables and figures for one component."""

    @staticmethod
    def component_path(component_name):
        """Return a filesystem-friendly component path segment.

        Parameters
        ----------
        component_name : object
            Component name to normalize.

        Returns
        -------
        str
            Trimmed name with whitespace replaced by underscores.
        """
        return re.sub(r"\s+", "_", str(component_name).strip())

    @staticmethod
    def clear_matching_files(folder, pattern):
        """Remove generated files in ``folder`` that match ``pattern``.

        Parameters
        ----------
        folder : pathlib.Path
            Existing output directory to inspect.
        pattern : str
            Glob pattern identifying stale generated files.
        """
        for stale_file in folder.glob(pattern):
            stale_file.unlink()

    def selected_fatigue_dlcs(self):
        """Return fatigue DLCs retained by the report-wide DLC selection."""
        fatigue_dlcs = normalize_dlc_filter(
            self.report.fatigue_dlc_list) or []
        selected_dlcs = normalize_dlc_filter(self.report.dlcs)
        if selected_dlcs is None:
            return fatigue_dlcs
        # Intersect with the configured fatigue list instead of forwarding the
        # generic filter: non-fatigue DLCs must not enter damage summaries.
        selected = set(selected_dlcs)
        return [dlc for dlc in fatigue_dlcs if dlc in selected]

    def build_latex_subsection(self, component_dir, relative_to=None):
        """Build this section's LaTeX content from its generated files."""
        # Import here to keep the shared section base independent while modules
        # are loaded.  ``dlb_latex`` provides the common LaTeX assembly logic.
        from wetb.dlb.report_generator.dlb_latex import build_report_subsection

        component_dir = Path(component_dir)
        table_folder = (
            component_dir / self.latex_table_subdirectory
            if self.latex_table_subdirectory else None
        )
        figure_folder = (
            component_dir / self.latex_figure_subdirectory
            if self.latex_figure_subdirectory else None
        )
        return build_report_subsection(
            self.latex_title,
            table_folder,
            figure_folder,
            relative_to=relative_to,
            figure_renderer=self.render_latex_figures,
            figure_intro=self.latex_figure_intro(),
        )

    def render_latex_figures(self, figure_folder, relative_to=None):
        """Return normal figure includes; sections override special layouts."""
        from wetb.dlb.report_generator.dlb_latex import include_png_files

        return include_png_files(figure_folder, relative_to)

    def latex_figure_intro(self):
        """Return optional LaTeX placed before this section's figures."""
        return ""

