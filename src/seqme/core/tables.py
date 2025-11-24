from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


def show(
    df: pd.DataFrame,
    *,
    n_decimals: int | list[int] = 2,
    color: str | None = "#68d6bc",
    color_style: Literal["solid", "gradient", "bar"] = "solid",
    notation: Literal["decimals", "exponent"] | list[Literal["decimals", "exponent"]] = "decimals",
    na_value: str = "-",
    show_arrow: bool = True,
    level: int = 0,
    hline_level: int | None = None,
    caption: str | None = None,
) -> Styler:
    """Display a metric dataframe as a styled table.

    Render a styled DataFrame that:
        - Combines 'value' and 'deviation' into "value ± deviation".
        - Highlights the best metric per column with color.
        - Underlines the second-best metric per column.
        - Arrows indicate maximize (↑) or minimize (↓).
        - Vertical divider between columns.

    Args:
        df: DataFrame with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], attributed with 'objective'.
        n_decimals: Decimal precision for formatting. Value is rounded. Deviation is rounded up.
        color: Color for highlighting best scores. If ``None``, no coloring.
        color_style: Style of the coloring. Ignored if color is ``None``.
        notation: Whether to use scientific notation (exponent) or fixed-point notation (decimals).
        na_value: Text to show for cells with no metric value, i.e., cells with NaN values.
        show_arrow: Whether to include the objective arrow in the column names.
        level: The tuple index-names level to consider as a group.
        hline_level: When to add horizontal lines seperaing model names. If ``None``, add horizontal lines at the first level if more than 1 level.
        caption: Bottom caption text. If ``None``, no caption is added.

    Returns:
        Styler: pandas Styler object.
    """

    def format_cell(
        val: float,
        dev: float,
        n_decimals: int,
        notation: Literal["decimals", "exponent"],
        no_value: str,
    ) -> str:
        notation_formatters = {"decimals": "f", "exponent": "e"}
        suffix_notation = notation_formatters[notation]

        if pd.isna(val):
            return no_value

        fval = _format_precision(val, n_decimals, suffix_notation)
        if pd.isna(dev):
            return fval

        fdev = _format_precision(dev, n_decimals, suffix_notation)
        return f"{fval}±{fdev}"

    def _fraction(val: float, min_value: float, max_value: float, objective: str) -> float:
        if pd.isna(val):
            return 0.0
        if max_value <= min_value:
            return 1.0
        t = (val - min_value) / (max_value - min_value)
        return 1 - t if objective == "minimize" else t

    def decorate_solid(idx: int, metric: str, df: pd.DataFrame, is_best: bool, is_second_best: bool) -> str:
        fmts = []
        if is_best:
            if color:
                fmts += [f"background-color:{color}"]
            fmts += ["font-weight:bold"]
        if is_second_best:
            fmts += ["text-decoration:underline"]
        return "; ".join(fmts)

    def decorate_gradient(idx: int, metric: str, df: pd.DataFrame, is_best: bool, is_second_best: bool) -> str:
        def gradient_lerp(hex_color1: str, hex_color2: str, t: float) -> str:
            cmap = mpl.colors.LinearSegmentedColormap.from_list(None, [hex_color1, hex_color2])
            return mpl.colors.to_hex(cmap(t), keep_alpha=True)

        fmts = []
        if color:
            objective = df.attrs["objective"][metric]
            values = df[(metric, "value")]
            frac = _fraction(values.at[idx], values.min(), values.max(), objective)
            gradient = gradient_lerp(f"{color}00", f"{color}ff", frac)
            fmts += [f"background-color:{gradient}"]

        if is_best:
            fmts += ["font-weight:bold"]
        if is_second_best:
            fmts += ["text-decoration:underline"]
        return "; ".join(fmts)

    def decorate_bar(idx: int, metric: str, df: pd.DataFrame, is_best: bool, is_second_best: bool) -> str:
        fmts = []
        if color:
            objective = df.attrs["objective"][metric]
            values = df[(metric, "value")]
            frac = _fraction(values.at[idx], values.min(), values.max(), objective)
            if frac > 0:
                width = f"{frac * 100:.1f}%"
                fmts += [f"background: linear-gradient(90deg, {color} {width}, transparent {width})"]

        if is_best:
            fmts += ["font-weight:bold"]
        if is_second_best:
            fmts += ["text-decoration:underline"]
        return "; ".join(fmts)

    def decorate_col(col_series: pd.Series, metric: str, fn: Callable, df: pd.DataFrame) -> list[str]:
        best_indices, second_best_indices = _get_top_indices(df, metric)
        return [fn(idx, metric, df, idx in best_indices, idx in second_best_indices) for idx in col_series.index]

    def get_changing_rows_iloc(indices: pd.Index, hline_level: int) -> list[int]:
        level_names = [idx[:hline_level] for idx in indices]
        changing_rows = []
        prev = None
        for i, v in enumerate(level_names):
            if i != 0 and v != prev:
                changing_rows.append(i)  # i is 0-based index into dataframe rows
            prev = v
        return changing_rows

    n_metrics = df.shape[1] // 2
    n_decimals = [n_decimals] * n_metrics if isinstance(n_decimals, int) else n_decimals
    notation = [notation] * n_metrics if isinstance(notation, str) else notation

    if len(n_decimals) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} decimals, got {len(n_decimals)}. Provide a single int or a list matching the number of metrics."
        )

    if len(notation) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} notations, got {len(notation)}. Provide a single int or a list matching the number of metrics."
        )

    if color is not None:
        if not mpl.colors.is_color_like(color):
            raise ValueError(f"Invalid color: {color}")
        color = mpl.colors.to_hex(color)

    if level >= df.index.nlevels or level < 0:
        raise ValueError(f"Level should be in range [0;{df.index.nlevels - 1}].")

    df = _round_dataframe(df, n_decimals)

    arrows = {"maximize": "↑", "minimize": "↓"}
    metrics = pd.unique(df.columns.get_level_values(0)).tolist()

    objectives = df.attrs["objective"]
    df_styled = pd.DataFrame(index=df.index)
    for i, m in enumerate(metrics):
        vals, devs = df[(m, "value")], df[(m, "deviation")]
        arrow = arrows[objectives[m]]
        col_name = f"{m}{arrow}" if show_arrow else m
        df_styled[col_name] = [
            format_cell(val, dev, n_decimals[i], notation[i], na_value) for val, dev in zip(vals, devs, strict=True)
        ]

    decorators = {"solid": decorate_solid, "gradient": decorate_gradient, "bar": decorate_bar}
    decorator = decorators[color_style]

    styler = df_styled.style

    # Decorate columns based on a levels groups
    groups = defaultdict(list)
    for index in df.index:
        level_index = index[:level]
        groups[level_index].append(index)

    for group in groups.values():
        df_sub = df.loc[group]

        for col, metric in zip(styler.columns, metrics, strict=True):
            styler = styler.apply(
                partial(decorate_col, metric=metric, fn=decorator, df=df_sub),
                axis=0,
                subset=(df_sub.index, [col]),  # type: ignore
            )

    table_styles = [
        {"selector": "th.col_heading", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("border-right", "1px solid #ccc")]},
        {"selector": "th.row_heading", "props": [("border-right", "1px solid #ccc")]},
    ]

    if hline_level is None:
        hline_level = 1 if df.index.nlevels > 1 else 0

    if hline_level > df.index.nlevels or hline_level < 0:
        raise ValueError(f"Level should be in range [0;{df.index.nlevels}].")

    if hline_level > 0:
        rows_iloc = get_changing_rows_iloc(df.index, hline_level)
        for row_idx in rows_iloc:
            nth_child = row_idx + 1  # add CSS using tbody nth-child (nth-child is 1-based, so add 1)
            selector = f"tbody tr:nth-child({nth_child}) td, tbody tr:nth-child({nth_child}) th"
            table_styles += [({"selector": selector, "props": [("border-top", "1px solid #ccc")]})]

    if caption:
        styler = styler.set_caption(caption)
        table_styles += [{"selector": "caption", "props": [("caption-side", "bottom"), ("margin-top", "0.75em")]}]

    styler = styler.set_table_styles(table_styles, overwrite=False)  # type: ignore

    return styler


def to_latex(
    df: pd.DataFrame,
    path: str | Path,
    *,
    n_decimals: int | list[int] = 2,
    color: str | None = None,
    na_value: str = "-",
    show_arrow: bool = True,
    caption: str | None = None,
    label: str | None = "tbl:benchmark",
):
    """Export a metric dataframe to a LaTeX table.

    Args:
        df: DataFrame with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], attributed with 'objective'.
        path: Output filename, e.g., ``"./path/table.tex"``.
        n_decimals: Decimal precision for formatting. Value is rounded. Deviation is rounded up.
        color: Color for highlighting best scores. If ``None``, no coloring.
        na_value: Text to show for cells with no metric value, i.e., cells with NaN values.
        show_arrow: Whether to include the objective arrow in the column names.
        caption: Bottom caption text. If ``None``, no caption is added.
        label: Table label. Identifier used to reference the table. If ``None``, no label is added.
    """
    # @TODO: support multi-index rows + levels
    if df.index.nlevels != 1:
        raise ValueError("to_latex() does not support tuple sequence names.")

    if color is not None:
        if not mpl.colors.is_color_like(color):
            raise ValueError(f"Invalid color: {color}")
        color = mpl.colors.to_hex(color)

    n_metrics = df.shape[1] // 2
    n_decimals = [n_decimals] * n_metrics if isinstance(n_decimals, int) else n_decimals

    if len(n_decimals) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} decimals, got {len(n_decimals)}. Provide a single int or a list matching the number of metrics."
        )

    df = _round_dataframe(df, n_decimals)

    best_indices, second_best_indices = {}, {}
    metrics = pd.unique(df.columns.get_level_values(0)).tolist()
    for m in metrics:
        best_indices[m], second_best_indices[m] = _get_top_indices(df, m)

    objectives = df.attrs["objective"]
    arrows = {"maximize": "↑", "minimize": "↓"}

    col_names = list(df.columns.get_level_values(0).unique())
    n_cols = len(col_names)
    n_row_levels = df.index.nlevels
    n_cols_and_row_levels = n_row_levels + n_cols

    # LaTeX formatting

    class WriteBuffer:
        def __init__(self):
            self.content = ""
            self.indent_level = 0

        def append(self, c: str):
            self.content += c

        def line(self, c: str):
            content = "\t" * self.indent_level + c + "\\\\" + "\n"
            self.append(content)

        def inline(self, c: str):
            content = "\t" * self.indent_level + c + "\n"
            self.append(content)

        def indent(self):
            self.indent_level += 1

        def unindent(self):
            self.indent_level -= 1

        def dump(self) -> str:
            return self.content

    def table_header(metric: str) -> str:
        arrow = arrows[objectives[metric]]
        text = f"{metric} ({arrow})" if show_arrow else metric
        return f"\\textbf{{{text}}}"

    def to_row(columns: list[str]) -> str:
        return " & ".join(columns)

    buffer = WriteBuffer()
    buffer.inline("\\begin{table}[h]")
    buffer.indent()

    buffer.inline("\\centering")
    buffer.inline("\\resizebox{\\textwidth}{!}{")

    cols = "c" * n_cols_and_row_levels
    buffer.inline(f"\\begin{{tabular}}{{{cols}}}")
    buffer.indent()

    buffer.inline("\\toprule")

    row_headers = ["\\textbf{Method}"] + [table_header(metric) for metric in col_names]
    buffer.line(to_row(row_headers))

    buffer.inline("\\midrule")

    for row_name, row in df.iterrows():
        values = [row_name]
        for i, (val, dev) in enumerate(zip(row[::2], row[1::2], strict=True)):
            if pd.isna(val):
                values.append(na_value)
                continue

            col_name = col_names[i]
            n_decimal = n_decimals[i]

            fval = _format_precision(val, n_decimal)
            fdev = _format_precision(dev, n_decimal) if not pd.isna(dev) else None

            best = row_name in best_indices[col_name]
            second_best = row_name in second_best_indices[col_name]

            if best:
                v = f"\\mathbf{{{fval}}}"
                if fdev:
                    v += f" \\pm \\mathbf{{{fdev}}}"
                if color:
                    v = f"\\cellcolor[HTML]{{{color[1:]}}}{{{v}}}"
            elif second_best:
                v = f"{fval} \\pm {fdev}" if fdev else f"{fval}"
                v = f"\\underline{{{v}}}"
            else:
                v = f"{fval} \\pm {fdev}" if fdev else f"{fval}"

            values.append(f"${v}$")

        buffer.line(to_row(values))

    buffer.inline("\\bottomrule")

    buffer.unindent()
    buffer.inline("\\end{tabular}")

    buffer.inline("} %resizebox")

    if caption:
        buffer.inline(f"\\caption{{{caption}}}")

    if label:
        buffer.inline(f"\\label{{{label}}}")

    buffer.unindent()
    buffer.inline("\\end{table}")

    latex_code = buffer.dump()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_code)


def _get_top_indices(df: pd.DataFrame, metric: str) -> tuple[set[int], set[int]]:
    def top_indices_helper(top_two: pd.Series) -> tuple[set[int], set[int]]:
        if pd.isna(top_two.values[0]):
            return set(), set()

        # get all indices with the same value as the best value
        value1 = top_two.values[0]
        indices1 = top_two.index[top_two == value1].tolist()
        if len(indices1) >= 2:
            return set(indices1), set()

        if len(top_two) < 2 or pd.isna(top_two.values[1]):
            return set(indices1), set()

        # get all indices with the same value as the second best value
        value2 = top_two.values[1]
        indices2 = top_two.index[top_two == value2].tolist()

        return set(indices1), set(indices2)

    if "objective" not in df.attrs:
        raise ValueError("DataFrame must have an 'objective' attribute. Use 'sm.evaluate' to create the DataFrame.")

    objective = df.attrs["objective"][metric]
    vals = df[(metric, "value")]

    if objective == "maximize":
        best_cells = vals.nlargest(2, keep="all")
    elif objective == "minimize":
        best_cells = vals.nsmallest(2, keep="all")
    else:
        raise ValueError(f"Unknown objective '{objective}' for metric '{metric}'.")

    return top_indices_helper(best_cells)


def _round_dataframe(df: pd.DataFrame, n_decimals: list[int]) -> pd.DataFrame:
    df = df.copy()

    n_decimals = [d for d in n_decimals for _ in range(2)]
    for n_decimal, (col, series) in zip(n_decimals, df.items(), strict=True):
        if col[1] == "value":
            df[col] = _round_column(series, n_decimal)
        elif col[1] == "deviation":
            df[col] = _ceil_column(series, n_decimal)
        else:
            raise ValueError(f"Invalid multi-index column: {col}")
    return df


def _ceil_column(series: pd.Series, n: int) -> pd.Series:
    factor = 10**n
    return np.ceil(series * factor) / factor


def _round_column(series: pd.Series, n: int) -> pd.Series:
    factor = 10**n
    return np.round(series * factor) / factor


def _format_precision(val: float, n_decimals: int, suffix: str = "f") -> str:
    return f"{val:.{n_decimals}{suffix}}"
