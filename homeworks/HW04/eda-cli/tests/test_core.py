from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
    DatasetSummary,
    ColumnSummary,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# НОВЫЕ ТЕСТЫ ДЛЯ HW03

def test_compute_quality_flags_with_constant_column():
    """Тест для обнаружения константных колонок"""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["A", "B", "C", "D"],
        "constant_col": ["same", "same", "same", "same"],  # Константная колонка
        "mixed": ["x", "y", "x", "y"]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert "has_constant_columns" in flags
    assert flags["has_constant_columns"] is True
    assert "constant_columns" in flags
    assert "constant_col" in flags["constant_columns"]
    assert "constant_columns_count" in flags
    assert flags["constant_columns_count"] == 1


def test_compute_quality_flags_with_high_cardinality():
    """Тест для обнаружения высокой кардинальности"""
    # Создаем категориальный признак с высокой кардинальностью
    df = pd.DataFrame({
        "id": range(150),  # 150 строк
        "high_card_col": [f"value_{i}" for i in range(150)],  # 150 уникальных значений
        "low_card_col": ["A", "B"] * 75  # Только 2 уникальных значения
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Порог по умолчанию 100
    flags = compute_quality_flags(summary, missing_df, high_cardinality_threshold=100)
    
    assert "has_high_cardinality_categoricals" in flags
    assert flags["has_high_cardinality_categoricals"] is True
    assert "high_cardinality_columns" in flags
    assert len(flags["high_cardinality_columns"]) == 1
    assert flags["high_cardinality_columns"][0]["name"] == "high_card_col"
    assert flags["high_cardinality_columns"][0]["unique"] == 150
    assert "high_cardinality_count" in flags
    assert flags["high_cardinality_count"] == 1
    assert "high_cardinality_threshold" in flags
    assert flags["high_cardinality_threshold"] == 100


def test_compute_quality_flags_custom_threshold():
    """Тест с пользовательским порогом для высокой кардинальности"""
    df = pd.DataFrame({
        "id": range(50),
        "col1": [f"val_{i}" for i in range(50)],  # 50 уникальных значений
        "col2": ["A", "B"] * 25  # 2 уникальных значения
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Порог 30 - col1 должна считаться высокой кардинальностью
    flags_low_threshold = compute_quality_flags(
        summary, missing_df, high_cardinality_threshold=30
    )
    assert flags_low_threshold["has_high_cardinality_categoricals"] is True
    assert len(flags_low_threshold["high_cardinality_columns"]) == 1
    
    # Порог 100 - ни одна колонка не должна считаться высокой кардинальностью
    flags_high_threshold = compute_quality_flags(
        summary, missing_df, high_cardinality_threshold=100
    )
    assert flags_high_threshold["has_high_cardinality_categoricals"] is False
    assert len(flags_high_threshold["high_cardinality_columns"]) == 0


def test_quality_score_with_new_heuristics():
    """Тест, что новые эвристики влияют на quality_score"""
    df = pd.DataFrame({
        "id": range(10),
        "constant_col": ["same"] * 10,
        "high_card_col": [str(i) for i in range(10)]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, high_cardinality_threshold=5)
    
    assert flags["has_constant_columns"] is True
    assert flags["has_high_cardinality_categoricals"] is True
    
    # Оценка должна быть ниже из-за константных колонок и высокой кардинальности
    score_with_issues = flags["quality_score"]
    
    # Создаем "хороший" датасет без проблем
    df_good = pd.DataFrame({
        "id": range(100),
        "feature1": np.random.randn(100),
        "feature2": np.random.choice(["A", "B", "C"], 100)
    })
    
    summary_good = summarize_dataset(df_good)
    missing_df_good = missing_table(df_good)
    flags_good = compute_quality_flags(summary_good, missing_df_good)
    
    # Оценка хорошего датасета должна быть выше
    assert flags_good["quality_score"] > score_with_issues


def test_dataset_summary_to_dict():
    """Тест сериализации DatasetSummary в словарь"""
    df = _sample_df()
    summary = summarize_dataset(df)
    
    summary_dict = summary.to_dict()
    
    assert "n_rows" in summary_dict
    assert "n_cols" in summary_dict
    assert "columns" in summary_dict
    assert len(summary_dict["columns"]) == 3
    
    # Проверяем, что каждая колонка сериализуется
    for col_dict in summary_dict["columns"]:
        assert "name" in col_dict
        assert "dtype" in col_dict
        assert "is_numeric" in col_dict


def test_column_summary_to_dict():
    """Тест сериализации ColumnSummary в словарь"""
    df = pd.DataFrame({"test_col": [1, 2, 3]})
    summary = summarize_dataset(df)
    column = summary.columns[0]
    
    column_dict = column.to_dict()
    
    assert column_dict["name"] == "test_col"
    assert "dtype" in column_dict
    assert "non_null" in column_dict
    assert column_dict["non_null"] == 3
    assert "is_numeric" in column_dict
    assert column_dict["is_numeric"] is True


# ТЕСТЫ ДЛЯ HW04 - HTTP API

def test_api_compatible_flags_structure():
    """Тест, что флаги совместимы с API (содержат нужные поля)"""
    df = _sample_df()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем наличие ключевых полей для API
    required_fields = [
        "too_few_rows",
        "too_many_columns", 
        "too_many_missing",
        "max_missing_share",
        "has_constant_columns",
        "has_high_cardinality_categoricals",
        "quality_score"
    ]
    
    for field in required_fields:
        assert field in flags, f"Отсутствует поле {field} во флагах"
    
    # Проверяем типы полей
    assert isinstance(flags["too_few_rows"], bool)
    assert isinstance(flags["too_many_columns"], bool)
    assert isinstance(flags["too_many_missing"], bool)
    assert isinstance(flags["has_constant_columns"], bool)
    assert isinstance(flags["has_high_cardinality_categoricals"], bool)
    assert isinstance(flags["quality_score"], float)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_flags_json_serializable():
    """Тест, что все флаги можно сериализовать в JSON"""
    df = _sample_df()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Пробуем сериализовать в JSON
    try:
        json_str = json.dumps(flags, ensure_ascii=False)
        parsed_back = json.loads(json_str)
        
        # Проверяем, что ключевые поля сохранились
        assert "quality_score" in parsed_back
        assert parsed_back["quality_score"] == flags["quality_score"]
    except (TypeError, ValueError) as e:
        assert False, f"Флаги не сериализуемы в JSON: {e}"


def test_compute_quality_flags_with_empty_dataset():
    """Тест обработки пустого датасета"""
    df = pd.DataFrame()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Не должно быть исключений
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем доступные поля (используем summary для n_rows и n_cols)
    assert summary.n_rows == 0
    assert summary.n_cols == 0
    assert flags["too_few_rows"] is True
    assert flags["has_constant_columns"] is False
    assert flags["has_high_cardinality_categoricals"] is False


def test_compute_quality_flags_edge_cases():
    """Тест граничных случаев"""
    # Датфрейм только с пропусками
    df_all_nan = pd.DataFrame({
        "col1": [None, None, None],
        "col2": [None, None, None]
    })
    
    summary = summarize_dataset(df_all_nan)
    missing_df = missing_table(df_all_nan)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags["max_missing_share"] == 1.0
    assert flags["too_many_missing"] is True
    assert flags["quality_score"] == 0.0
    
    # Датфрейм с одной строкой
    df_one_row = pd.DataFrame({
        "col1": [1],
        "col2": ["A"]
    })
    
    summary = summarize_dataset(df_one_row)
    missing_df = missing_table(df_one_row)
    flags = compute_quality_flags(summary, missing_df)
    
    assert flags["too_few_rows"] is True
    # Для одной строки все колонки являются константными
    assert flags["has_constant_columns"] is True


def test_csv_file_compatibility():
    """Тест совместимости с CSV файлами (для API)"""
    # Создаем временный CSV файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35]
        })
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Читаем CSV как это делает API
        df_from_csv = pd.read_csv(temp_path)
        
        # Проверяем, что данные корректны
        assert df_from_csv.shape == (3, 3)
        assert "id" in df_from_csv.columns
        assert "name" in df_from_csv.columns
        assert "age" in df_from_csv.columns
        
        # Проверяем через нашу логику
        summary = summarize_dataset(df_from_csv)
        missing_df = missing_table(df_from_csv)
        flags = compute_quality_flags(summary, missing_df)
        
        # Используем summary для получения n_rows и n_cols
        assert summary.n_rows == 3
        assert summary.n_cols == 3
        assert flags["has_constant_columns"] is False  # Нет константных колонок
        
    finally:
        # Удаляем временный файл
        Path(temp_path).unlink()


def test_top_categories_with_custom_k():
    """Тест top_categories с разными значениями top_k"""
    df = pd.DataFrame({
        "category": ["A", "B", "A", "C", "B", "A", "D", "E", "F", "G"]
    })
    
    # top_k=3
    top_cats_3 = top_categories(df, max_columns=5, top_k=3)
    assert "category" in top_cats_3
    assert len(top_cats_3["category"]) == 3
    
    # top_k=5  
    top_cats_5 = top_categories(df, max_columns=5, top_k=5)
    assert len(top_cats_5["category"]) == 5
    
    # Проверяем, что возвращаются самые частые значения
    value_counts = df["category"].value_counts()
    most_common = value_counts.index[0]
    assert top_cats_3["category"].iloc[0]["value"] == most_common


def test_flags_structure_for_api_response():
    """Тест структуры флагов для API ответа"""
    df = _sample_df()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Для API ответа нужно проверять структуру
    api_response_like = {
        "flags": flags,
        "dataset_shape": {
            "n_rows": summary.n_rows,  # Берем из summary, а не из flags
            "n_cols": summary.n_cols   # Берем из summary, а не из flags
        },
        "latency_ms": 0.0,
        "filename": "test.csv"
    }
    
    # Проверяем структуру
    assert "flags" in api_response_like
    assert "dataset_shape" in api_response_like
    assert "latency_ms" in api_response_like
    assert "filename" in api_response_like
    
    # Проверяем вложенные поля
    assert "n_rows" in api_response_like["dataset_shape"]
    assert "n_cols" in api_response_like["dataset_shape"]
    
    # Проверяем, что новые флаги из HW03 присутствуют
    assert "has_constant_columns" in api_response_like["flags"]
    assert "has_high_cardinality_categoricals" in api_response_like["flags"]


def test_quality_score_range():
    """Тест, что quality_score всегда в диапазоне [0, 1]"""
    test_cases = [
        # (датасет, описание)
        (pd.DataFrame({"col": [1, 2, 3]}), "Хороший датасет"),
        (pd.DataFrame({"col": [None, None, None]}), "Только пропуски"),
        (pd.DataFrame({"col": range(1000)}), "Много строк"),
        (pd.DataFrame({f"col{i}": [1] for i in range(200)}), "Много колонок"),
    ]
    
    for df, description in test_cases:
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)
        
        assert 0.0 <= flags["quality_score"] <= 1.0, \
            f"quality_score вне диапазона для {description}: {flags['quality_score']}"


def test_export_import_cycle():
    """Тест цикла экспорт-импорт данных через JSON"""
    df = _sample_df()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Экспортируем в JSON
    json_data = json.dumps({
        "summary": summary.to_dict(),
        "flags": flags
    }, ensure_ascii=False)
    
    # Импортируем обратно
    imported = json.loads(json_data)
    
    # Проверяем структуру
    assert "summary" in imported
    assert "flags" in imported
    
    # Проверяем ключевые поля
    assert imported["summary"]["n_rows"] == 4
    assert imported["summary"]["n_cols"] == 3
    assert imported["flags"]["quality_score"] == flags["quality_score"]
