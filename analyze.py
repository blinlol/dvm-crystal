from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_dataset(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	return df


def print_basic_stats(df: pd.DataFrame) -> None:
	print("=== Общая информация ===")
	print(f"Строки: {len(df):,}")
	print(f"Столбцы: {df.shape[1]:,}")
	print("\n=== Типы данных ===")
	print(df.dtypes.sort_values())

	print("\n=== Пропуски (топ-20) ===")
	missing = df.isna().sum().sort_values(ascending=False)
	print(missing.head(20))

	dup_count = df.duplicated().sum()
	print(f"\nДубликаты строк: {dup_count:,}")

	numeric_cols = df.select_dtypes(include="number").columns
	numeric_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
	non_finite = (~np.isfinite(numeric_df)).sum().sum()
	if non_finite > 0:
		print(f"\nНайдено нечисловых значений в числовых колонках (NaN/inf): {non_finite:,}")
	print("\n=== Числовая статистика ===")
	print(numeric_df.describe().T)

	cat_cols = df.select_dtypes(include=["object", "category"]).columns
	if len(cat_cols) > 0:
		print("\n=== Категориальные признаки (value_counts, топ-10) ===")
		for col in cat_cols:
			print(f"\n[{col}]")
			print(df[col].value_counts(dropna=False).head(10))


def save_plot(fig: plt.Figure, out_dir: Path, name: str) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / name
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def generate_plots(df: pd.DataFrame, out_dir: Path) -> None:
	sns.set_theme(style="whitegrid")

	numeric_cols = df.select_dtypes(include="number").columns
	target_col = "parallel_execution_time"

	if target_col in df.columns:
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.histplot(df[target_col].dropna(), bins=40, kde=True, ax=ax)
		ax.set_title("Распределение parallel_execution_time")
		save_plot(fig, out_dir, "target_distribution.png")

	if len(numeric_cols) > 1:
		corr = df[numeric_cols].corr(numeric_only=True)
		fig, ax = plt.subplots(figsize=(12, 10))
		sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
		ax.set_title("Корреляции числовых признаков")
		save_plot(fig, out_dir, "correlation_heatmap.png")

	if {"launch_total_processors", target_col}.issubset(df.columns):
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.scatterplot(
			data=df,
			x="launch_total_processors",
			y=target_col,
			hue="program_name" if "program_name" in df.columns else None,
			ax=ax,
		)
		ax.set_title("Время выполнения vs число процессоров")
		save_plot(fig, out_dir, "time_vs_processors.png")

	if {"launch_threads", target_col}.issubset(df.columns):
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.boxplot(data=df, x="launch_threads", y=target_col, ax=ax)
		ax.set_title("Время выполнения по числу потоков")
		save_plot(fig, out_dir, "time_by_threads.png")

	if {"program_name", target_col}.issubset(df.columns):
		fig, ax = plt.subplots(figsize=(10, 5))
		sns.boxplot(data=df, x="program_name", y=target_col, ax=ax)
		ax.set_title("Время выполнения по программам")
		ax.tick_params(axis="x", rotation=45)
		save_plot(fig, out_dir, "time_by_program.png")

	if "normalized_parallel_time" in df.columns:
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.histplot(df["normalized_parallel_time"].dropna(), bins=40, kde=True, ax=ax)
		ax.set_title("Распределение normalized_parallel_time")
		save_plot(fig, out_dir, "normalized_parallel_time_distribution.png")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Анализ feature_dataset.csv")
	parser.add_argument(
		"--csv",
		type=Path,
		default=Path("data/processed/feature_dataset.csv"),
		help="Путь к CSV файлу",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path("reports/plots"),
		help="Папка для сохранения графиков",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = load_dataset(args.csv)
	print_basic_stats(df)
	generate_plots(df, args.out)
	print(f"\nГрафики сохранены в: {args.out.resolve()}")


if __name__ == "__main__":
	main()
