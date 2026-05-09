from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from .metrics import pct_change


Curve = list[tuple[date, float]]


def write_simulation_artifacts(
    strategy_curve: Curve,
    benchmark_curve: Curve,
    output_dir: Path = Path("reports"),
    label: str = "simulation",
) -> tuple[Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{label}_{timestamp}.csv"
    svg_path = output_dir / f"{label}_{timestamp}.svg"
    png_path = output_dir / f"{label}_{timestamp}.png"
    _write_csv(csv_path, strategy_curve, benchmark_curve)
    _write_svg(svg_path, strategy_curve, benchmark_curve)
    rendered_png = _write_png(png_path, strategy_curve, benchmark_curve)
    return csv_path, svg_path, rendered_png


def write_daily_return_artifacts(
    strategy_curve: Curve,
    output_dir: Path = Path("reports"),
    label: str = "daily_returns",
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{label}_{timestamp}.csv"
    png_path = output_dir / f"{label}_{timestamp}.png"
    _write_daily_return_csv(csv_path, strategy_curve)
    rendered_png = _write_daily_return_png(png_path, strategy_curve)
    return csv_path, rendered_png


def write_model_comparison_artifacts(
    curves: dict[str, Curve],
    output_dir: Path = Path("reports"),
    label: str = "model_comparison",
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{label}_{timestamp}.csv"
    png_path = output_dir / f"{label}_{timestamp}.png"
    _write_model_comparison_csv(csv_path, curves)
    rendered_png = _write_model_comparison_png(png_path, curves)
    return csv_path, rendered_png


def _write_csv(path: Path, strategy_curve: Curve, benchmark_curve: Curve) -> None:
    benchmark_by_date = {item_date: value for item_date, value in benchmark_curve}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "strategy_equity", "benchmark_equity"])
        for item_date, strategy_value in strategy_curve:
            writer.writerow([item_date.isoformat(), f"{strategy_value:.2f}", f"{benchmark_by_date.get(item_date, 0):.2f}"])


def _write_daily_return_csv(path: Path, strategy_curve: Curve) -> None:
    returns = pct_change([value for _, value in strategy_curve])
    dates = [item_date for item_date, _ in strategy_curve][1:]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "daily_return"])
        for item_date, daily_return in zip(dates, returns):
            writer.writerow([item_date.isoformat(), f"{daily_return:.8f}"])


def _write_model_comparison_csv(path: Path, curves: dict[str, Curve]) -> None:
    all_dates = sorted({item_date for curve in curves.values() for item_date, _ in curve})
    values_by_model = {
        name: {item_date: value for item_date, value in curve}
        for name, curve in curves.items()
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", *curves.keys()])
        for item_date in all_dates:
            writer.writerow(
                [item_date.isoformat()]
                + [f"{values_by_model[name].get(item_date, 0):.2f}" for name in curves]
            )


def _write_svg(path: Path, strategy_curve: Curve, benchmark_curve: Curve) -> None:
    width = 1200
    height = 680
    margin_left = 86
    margin_right = 44
    margin_top = 64
    margin_bottom = 88
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    curves = [("전략 총자산", strategy_curve, "#2563eb"), ("SPY 보유 총자산", benchmark_curve, "#dc2626")]
    all_points = [point for _, curve, _ in curves for point in curve]
    if not all_points:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return

    min_date = min(point[0] for point in all_points).toordinal()
    max_date = max(point[0] for point in all_points).toordinal()
    min_value = min(point[1] for point in all_points) * 0.96
    max_value = max(point[1] for point in all_points) * 1.04
    if min_value == max_value:
        max_value += 1

    def x_scale(item_date: date) -> float:
        if max_date == min_date:
            return margin_left
        return margin_left + ((item_date.toordinal() - min_date) / (max_date - min_date)) * plot_width

    def y_scale(value: float) -> float:
        return margin_top + (1 - ((value - min_value) / (max_value - min_value))) * plot_height

    def polyline(curve: Curve) -> str:
        return " ".join(f"{x_scale(item_date):.2f},{y_scale(value):.2f}" for item_date, value in curve)

    grid_lines = []
    for idx in range(6):
        y = margin_top + idx * plot_height / 5
        value = max_value - idx * (max_value - min_value) / 5
        grid_lines.append(
            f"<line x1=\"{margin_left}\" y1=\"{y:.2f}\" x2=\"{width - margin_right}\" y2=\"{y:.2f}\" stroke=\"#e5e7eb\"/>"
            f"<text x=\"{margin_left - 12}\" y=\"{y + 4:.2f}\" text-anchor=\"end\" font-size=\"14\" fill=\"#475569\">${value:,.0f}</text>"
        )

    legend = []
    for idx, (name, curve, color) in enumerate(curves):
        if not curve:
            continue
        final_value = curve[-1][1]
        y = margin_top + idx * 28
        legend.append(
            f"<rect x=\"{width - 300}\" y=\"{y - 13}\" width=\"18\" height=\"4\" fill=\"{color}\"/>"
            f"<text x=\"{width - 272}\" y=\"{y - 8}\" font-size=\"16\" fill=\"#0f172a\">{name}: ${final_value:,.0f}</text>"
        )

    start_label = min(point[0] for point in all_points).isoformat()
    end_label = max(point[0] for point in all_points).isoformat()
    lines = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        "<text x=\"86\" y=\"38\" font-size=\"26\" font-family=\"Malgun Gothic, Arial, sans-serif\" font-weight=\"700\" fill=\"#0f172a\">전략 시뮬레이션 총자산 추이</text>",
        "<text x=\"86\" y=\"60\" font-size=\"14\" font-family=\"Malgun Gothic, Arial, sans-serif\" fill=\"#64748b\">파란색: 전략 총자산, 빨간색: SPY 단순 보유 총자산</text>",
        *grid_lines,
        f"<line x1=\"{margin_left}\" y1=\"{margin_top}\" x2=\"{margin_left}\" y2=\"{height - margin_bottom}\" stroke=\"#94a3b8\"/>",
        f"<line x1=\"{margin_left}\" y1=\"{height - margin_bottom}\" x2=\"{width - margin_right}\" y2=\"{height - margin_bottom}\" stroke=\"#94a3b8\"/>",
        f"<text x=\"{margin_left}\" y=\"{height - 42}\" font-size=\"14\" fill=\"#475569\">{start_label}</text>",
        f"<text x=\"{width - margin_right}\" y=\"{height - 42}\" text-anchor=\"end\" font-size=\"14\" fill=\"#475569\">{end_label}</text>",
        *legend,
    ]
    for name, curve, color in curves:
        if curve:
            lines.append(
                f"<polyline points=\"{polyline(curve)}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"3\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>"
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_png(path: Path, strategy_curve: Curve, benchmark_curve: Curve) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not strategy_curve:
        return None

    _configure_korean_font()
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=140)
    strategy_dates = [item_date for item_date, _ in strategy_curve]
    strategy_values = [value for _, value in strategy_curve]
    ax.plot(strategy_dates, strategy_values, color="#2563eb", linewidth=2.4, label="전략 총자산")

    if benchmark_curve:
        benchmark_dates = [item_date for item_date, _ in benchmark_curve]
        benchmark_values = [value for _, value in benchmark_curve]
        ax.plot(benchmark_dates, benchmark_values, color="#dc2626", linewidth=2.0, label="SPY 보유 총자산")

    ax.set_title("전략 시뮬레이션 총자산 추이", fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("총자산 ($)")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_daily_return_png(path: Path, strategy_curve: Curve) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    returns = pct_change([value for _, value in strategy_curve])
    dates = [item_date for item_date, _ in strategy_curve][1:]
    if not returns:
        return None

    _configure_korean_font()
    rolling: list[float | None] = []
    for idx in range(len(returns)):
        if idx < 19:
            rolling.append(None)
        else:
            rolling.append(sum(returns[idx - 19 : idx + 1]) / 20)

    colors = ["#2563eb" if item >= 0 else "#dc2626" for item in returns]
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=140)
    ax.bar(dates, [item * 100 for item in returns], color=colors, width=1.0, alpha=0.70, label="일별 수익률")
    ax.plot(
        dates,
        [item * 100 if item is not None else None for item in rolling],
        color="#111827",
        linewidth=1.8,
        label="20일 평균",
    )
    ax.axhline(0, color="#0f172a", linewidth=0.8)
    ax.set_title("일별 수익률 시뮬레이션", fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("일별 수익률 (%)")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_model_comparison_png(path: Path, curves: dict[str, Curve]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not curves:
        return None

    _configure_korean_font()
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c"]
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=140)
    for idx, (name, curve) in enumerate(curves.items()):
        if not curve:
            continue
        dates = [item_date for item_date, _ in curve]
        values = [value for _, value in curve]
        ax.plot(dates, values, linewidth=2.1, label=name, color=colors[idx % len(colors)])

    ax.set_title("홀드아웃 모델별 총자산 비교", fontsize=16, fontweight="bold", loc="left")
    ax.set_ylabel("총자산 ($)")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _configure_korean_font() -> None:
    try:
        import matplotlib
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        candidates = [
            "Malgun Gothic",
            "Noto Sans KR",
            "Gulim",
            "Batang",
            "AppleGothic",
        ]
        selected = next((font for font in candidates if font in available), "DejaVu Sans")
        matplotlib.rcParams["font.family"] = selected
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        return
