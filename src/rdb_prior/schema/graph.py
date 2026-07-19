"""Deterministic Graphviz artifacts for physical schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape as html_escape
import logging
from pathlib import Path
import re
import shutil
import subprocess

from rdb_prior.compilation.model import (
    ColumnKind,
    PhysicalSchema,
    PhysicalTable,
)
from rdb_prior.schema.spec import TableRole


_LOGGER = logging.getLogger(__name__)
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RENDER_FORMATS = frozenset({"png", "svg", "pdf"})
_ROLE_COLORS = {
    TableRole.ENTITY: "#DCEBFA",
    TableRole.EVENT: "#FCE1C3",
    TableRole.LOOKUP: "#DDEFD8",
    TableRole.BRIDGE: "#E8DDF5",
    TableRole.DETAIL: "#F5E9B8",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaGraphConfig:
    """Controls DOT creation and optional Graphviz image rendering."""

    write_dot: bool = True
    render_format: str | None = None
    graphviz_command: str = "dot"
    include_columns: bool = True
    include_role_metadata: bool = True

    def __post_init__(self) -> None:
        for name in ("write_dot", "include_columns", "include_role_metadata"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.render_format is not None:
            if not isinstance(self.render_format, str):
                raise TypeError("render_format must be a string or None")
            normalized = self.render_format.strip().lower()
            if normalized not in _RENDER_FORMATS:
                allowed = ", ".join(sorted(_RENDER_FORMATS))
                raise ValueError(f"render_format must be one of: {allowed}")
            object.__setattr__(self, "render_format", normalized)
        if not isinstance(self.graphviz_command, str) or not (
            self.graphviz_command.strip()
        ):
            raise ValueError("graphviz_command must not be empty")
        object.__setattr__(
            self,
            "graphviz_command",
            self.graphviz_command.strip(),
        )
        if self.render_format is not None and not self.write_dot:
            raise ValueError("write_dot must be enabled when rendering a graph")

    def to_dict(self) -> dict[str, object]:
        return {
            "write_dot": self.write_dot,
            "render_format": self.render_format,
            "graphviz_command": self.graphviz_command,
            "include_columns": self.include_columns,
            "include_role_metadata": self.include_role_metadata,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaGraphArtifacts:
    dot_path: Path | None = None
    image_path: Path | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaGraphArtifactWriter:
    """Write one DOT file per schema and optionally render an image."""

    output_root: Path
    config: SchemaGraphConfig = SchemaGraphConfig()
    overwrite: bool = False
    _graphviz_executable: str | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        if not isinstance(self.config, SchemaGraphConfig):
            raise TypeError("config must be SchemaGraphConfig")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        if self.config.render_format is not None:
            executable = shutil.which(self.config.graphviz_command)
            if executable is None:
                raise RuntimeError(
                    "Graphviz rendering was requested but command "
                    f"{self.config.graphviz_command!r} was not found. Install "
                    "Graphviz or set schema_graph.graphviz_command."
                )
            object.__setattr__(self, "_graphviz_executable", executable)

    @property
    def graph_directory(self) -> Path:
        return self.output_root / "schema_graphs"

    def commit(
        self,
        *,
        sample_id: str,
        schema: PhysicalSchema,
    ) -> SchemaGraphArtifacts:
        if not self.config.write_dot:
            return SchemaGraphArtifacts()
        if not isinstance(sample_id, str) or not _ARTIFACT_ID.fullmatch(
            sample_id
        ):
            raise ValueError("sample_id is not safe for an artifact filename")
        if not isinstance(schema, PhysicalSchema):
            raise TypeError("schema must be PhysicalSchema")

        dot_path = self.graph_directory / f"{sample_id}.dot"
        image_path = None
        if self.config.render_format is not None:
            image_path = self.graph_directory / (
                f"{sample_id}.{self.config.render_format}"
            )
        self._assert_writable(dot_path)
        if image_path is not None:
            self._assert_writable(image_path)

        self._write_dot(dot_path, physical_schema_to_dot(schema, self.config))
        if image_path is not None:
            self._render(dot_path, image_path)
        return SchemaGraphArtifacts(
            dot_path=dot_path,
            image_path=image_path,
        )

    def _assert_writable(self, path: Path) -> None:
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Artifact already exists: {path}; use overwrite=True"
            )

    @staticmethod
    def _write_dot(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)

    def _render(self, dot_path: Path, image_path: Path) -> None:
        executable = self._graphviz_executable
        render_format = self.config.render_format
        if executable is None or render_format is None:  # pragma: no cover
            raise RuntimeError("Graphviz renderer was not initialized")

        temporary_path = image_path.with_suffix(image_path.suffix + ".tmp")
        try:
            subprocess.run(
                [
                    executable,
                    f"-T{render_format}",
                    str(dot_path),
                    "-o",
                    str(temporary_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if not temporary_path.is_file():
                raise RuntimeError(
                    "Graphviz reported success but did not create "
                    f"{image_path.name}"
                )
            temporary_path.replace(image_path)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(
                f"Graphviz failed to render {dot_path.name}: {detail}"
            ) from error
        except OSError as error:
            raise RuntimeError(
                f"Cannot execute Graphviz command {executable!r}: {error}"
            ) from error
        finally:
            temporary_path.unlink(missing_ok=True)
        _LOGGER.debug(
            "schema graph rendered: dot=%s image=%s",
            dot_path,
            image_path,
        )


def physical_schema_to_dot(
    schema: PhysicalSchema,
    config: SchemaGraphConfig | None = None,
) -> str:
    """Serialize a physical schema as stable Graphviz DOT."""
    if not isinstance(schema, PhysicalSchema):
        raise TypeError("schema must be PhysicalSchema")
    options = config or SchemaGraphConfig()
    if not isinstance(options, SchemaGraphConfig):
        raise TypeError("config must be SchemaGraphConfig or None")

    lines = [
        "digraph schema {",
        "  graph [rankdir=LR, bgcolor=\"white\", pad=\"0.2\", "
        "nodesep=\"0.45\", ranksep=\"0.75\"];",
        "  node [shape=plain, fontname=\"Helvetica\"];",
        "  edge [color=\"#52606D\", fontname=\"Helvetica\", "
        "fontsize=\"9\", arrowsize=\"0.75\"];",
        f'  label="{_dot_escape(schema.schema_id)}";',
        "  labelloc=\"t\";",
        "  fontsize=\"12\";",
    ]
    for table in schema.tables:
        lines.extend(_table_node_lines(table, options))
    for foreign_key in schema.foreign_keys:
        child = schema.table(foreign_key.child_table_id)
        child_column = child.column(foreign_key.child_column_id)
        edge_label = (
            f"{_dot_escape(foreign_key.name)}: "
            f"{_dot_escape(child_column.name)}\\n"
            f"{_dot_escape(foreign_key.cardinality.value)} / "
            f"{_dot_escape(foreign_key.optionality.value)}"
        )
        lines.append(
            f'  "{_dot_escape(foreign_key.parent_table_id)}" -> '
            f'"{_dot_escape(foreign_key.child_table_id)}" '
            f'[label="{edge_label}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _table_node_lines(
    table: PhysicalTable,
    config: SchemaGraphConfig,
) -> list[str]:
    color = _ROLE_COLORS[table.role]
    details = table.table_id
    if config.include_role_metadata:
        details += f" | role={table.role.value} | rank={table.rank}"
    lines = [
        f'  "{_dot_escape(table.table_id)}" [label=<',
        "    <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" "
        "CELLPADDING=\"5\" COLOR=\"#7B8794\">",
        f'      <TR><TD COLSPAN="3" BGCOLOR="{color}"><B>'
        f"{_html(table.name)}</B></TD></TR>",
        '      <TR><TD COLSPAN="3" BGCOLOR="#F7F8FA"><FONT '
        f'POINT-SIZE="9">{_html(details)}</FONT></TD></TR>',
    ]
    if config.include_columns:
        for column in table.columns:
            marker = _column_marker(column.kind, column.unique)
            nullability = " nullable" if column.nullable else ""
            lines.append(
                "      <TR>"
                f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9">{marker}</FONT></TD>'
                f'<TD ALIGN="LEFT"><FONT FACE="Courier">{_html(column.name)}</FONT></TD>'
                f'<TD ALIGN="LEFT"><FONT POINT-SIZE="9">'
                f"{_html(column.data_type.value + nullability)}</FONT></TD>"
                "</TR>"
            )
    lines.extend(("    </TABLE>", "  >];"))
    return lines


def _column_marker(kind: ColumnKind, unique: bool) -> str:
    if kind is ColumnKind.PRIMARY_KEY:
        return "PK"
    if kind is ColumnKind.FOREIGN_KEY:
        return "FK"
    if unique:
        return "UQ"
    if kind is ColumnKind.TIME:
        return "TIME"
    return ""


def _dot_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _html(value: object) -> str:
    return html_escape(str(value), quote=True)


__all__ = [
    "SchemaGraphArtifactWriter",
    "SchemaGraphArtifacts",
    "SchemaGraphConfig",
    "physical_schema_to_dot",
]
