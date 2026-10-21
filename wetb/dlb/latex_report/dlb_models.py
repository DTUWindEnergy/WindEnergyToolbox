from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WeightParams:
    n_years: int
    Vin: float
    Vout: float
    Vr: float
    Vref: float
    Vstep: float

    @classmethod
    def from_mapping(cls, values):
        return cls(
            n_years=values["n_years"],
            Vin=values["Vin"],
            Vout=values["Vout"],
            Vr=values["Vr"],
            Vref=values["Vref"],
            Vstep=values["Vstep"],
        )


@dataclass
class ReportConfig:
    output_dir: Path
    cases: dict[str, Path]
    main_tex_filename: str
    report_title: str

    @classmethod
    def from_mapping(cls, values):
        cases = values.get("cases")
        if cases is None:
            cases = {"v1": values["postproc_dir"]}
        cases = {str(name): Path(path) for name, path in cases.items()}
        if len(cases) not in (1, 2):
            raise ValueError("REPORT_CONFIG['cases'] must contain either 1 or 2 cases.")

        return cls(
            output_dir=Path(values["output_dir"]),
            cases=cases,
            main_tex_filename=values["main_tex_filename"],
            report_title=values["report_title"],
        )


@dataclass
class AnalysisConfig:
    dlcs: object
    safety_factor_list: dict[str, float] | None
    fatigue_dlc_list: list[str]
    weight_params: WeightParams

    @classmethod
    def from_mapping(cls, values):
        return cls(
            dlcs=values.get("dlcs"),
            safety_factor_list=values.get("safety_factor_list"),
            fatigue_dlc_list=list(values["fatigue_dlc_list"]),
            weight_params=WeightParams.from_mapping(values["weight_params"]),
        )


@dataclass
class SensorMatch:
    name: object = None
    description: object = None

    @classmethod
    def from_input(cls, value):
        if isinstance(value, str):
            return cls(name=value, description=value)
        if isinstance(value, dict):
            return cls(
                name=value.get("name"),
                description=value.get("description"),
            )
        raise ValueError("Component match must be either a string or a dict.")

    def as_dict(self):
        return {
            key: value
            for key, value in {
                "name": self.name,
                "description": self.description,
            }.items()
            if value is not None
        }


@dataclass
class ComponentConfig:
    name: str
    label: str
    enabled_sections: list[str]
    fatigue_m: object
    match: SensorMatch
    ids: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, component_name, values, normalize_sections):
        sections = values.get("sections", values.get("enabled_sections"))
        if sections is None:
            raise ValueError(f"COMPONENTS['{component_name}'] must define sections.")

        fatigue_m = values.get("m", values.get("fatigue_m"))
        if fatigue_m is None:
            raise ValueError(f"COMPONENTS['{component_name}'] must define m.")

        return cls(
            name=component_name,
            label=values.get("label", component_name),
            enabled_sections=normalize_sections(sections),
            fatigue_m=fatigue_m,
            match=SensorMatch.from_input(values.get("match", values.get("tags", {}))),
        )

    def get(self, key, default=None):
        values = {
            "label": self.label,
            "enabled_sections": self.enabled_sections,
            "sections": self.enabled_sections,
            "fatigue_m": self.fatigue_m,
            "m": self.fatigue_m,
            "match": self.match.as_dict(),
            "ids": self.ids,
        }
        return values.get(key, default)


@dataclass
class DLBCase:
    name: str
    postproc_dir: Path
    datasets: dict = field(default_factory=dict)
