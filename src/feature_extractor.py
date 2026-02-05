import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, List, Any
import os
import json

logger = logging.getLogger(__name__)


class DVMHFeatureSpaceCreator:
    """Класс для создания признакового пространства DVMH программ"""
    
    def __init__(self, dvmh_processors: int = 1, config_path: str = "config/data_config.yaml"):
        """
        Инициализация создателя признакового пространства
        
        Args:
            dvmh_processors: Количество DVMH процессоров
            config_path: Путь к конфигурационному файлу
        """
        self.dvmh_processors = dvmh_processors
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")
        self.config = self._load_config(config_path)
        
        # Настройки из конфига
        feature_config = self.config.get('feature_extraction', {})
        self.normalization_enabled = feature_config.get('normalization', {}).get('enabled', True)
        self.normalization_method = feature_config.get('normalization', {}).get('method', 'standard')
        
        self.logger.info(f"DVMHFeatureSpaceCreator инициализирован с {dvmh_processors} процессорами")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Конфигурационный файл не найден: {config_path}")
            return {}
    
    def extract_static_program_features(self, program_info: Dict) -> Dict[str, float]:
        """
        Извлечение статических признаков программы
        
        Args:
            program_info: Информация о программе
            
        Returns:
            Dict: Словарь со статическими признаками
        """
        features = {}
        
        # Основные временные характеристики
        features['sequential_execution_time'] = program_info.get('sequential_execution_time_sec', 0)
        features['average_parallel_line_executions'] = program_info.get('average_parallel_line_executions', 0)
        features['parallel_execution_fraction'] = program_info.get('parallel_execution_fraction', 0)
        
        # Анализ массивов
        arrays_info = program_info.get('arrays_info', [])
        if arrays_info:
            features['arrays_count'] = len(arrays_info)
            
            total_elements = 0
            total_size_bytes = 0
            max_dimension = 0
            min_dimension = float('inf')
            dimension_list = []
            
            for array in arrays_info:
                elements = 1
                for dim in array['dimensions']:
                    elements *= dim
                    max_dimension = max(max_dimension, dim)
                    min_dimension = min(min_dimension, dim)
                    dimension_list.append(dim)
                
                size_bytes = elements * array['element_size_bytes']
                total_elements += elements
                total_size_bytes += size_bytes
            
            features['total_array_elements'] = total_elements
            features['total_array_size_bytes'] = total_size_bytes
            features['avg_array_size_bytes'] = total_size_bytes / len(arrays_info)
            features['max_array_dimension'] = max_dimension
            features['min_array_dimension'] = min_dimension if min_dimension != float('inf') else 0
            features['avg_array_dimension'] = np.mean(dimension_list) if dimension_list else 0
        else:
            for key in ['arrays_count', 'total_array_elements', 'total_array_size_bytes', 
                        'avg_array_size_bytes', 'max_array_dimension', 'min_array_dimension', 
                        'avg_array_dimension']:
                features[key] = 0
        
        # Анализ директив
        directives = program_info.get('directives', {})
        
        # Shadow renew директивы
        shadow_renew_count = 0
        shadow_renew_arrays = set()
        if 'shadow_renew' in directives and directives['shadow_renew'] is not None:
            shadow_renew_values = directives['shadow_renew']
            shadow_renew_count = len(shadow_renew_values)
            for sr in shadow_renew_values:
                if 'array_id' in sr:
                    shadow_renew_arrays.add(sr['array_id'])
        
        features['shadow_renew_count'] = shadow_renew_count
        features['shadow_renew_unique_arrays'] = len(shadow_renew_arrays)
        
        # Across директивы
        if 'across' in directives and directives['across'] is not None:
            across_ops = directives['across']
            features['across_count'] = len(across_ops)
            
            total_comm_width = 0
            max_comm_width = 0
            across_patterns = {}
            
            for op in across_ops:
                for width in op['width']:
                    width_sum = sum(width)
                    total_comm_width += width_sum
                    max_comm_width = max(max_comm_width, width_sum)
                
                pattern = op.get('communication_pattern', 'unknown')
                across_patterns[pattern] = across_patterns.get(pattern, 0) + 1
            
            features['across_total_width'] = total_comm_width
            features['across_max_width'] = max_comm_width
            features['across_avg_width'] = total_comm_width / len(across_ops) if across_ops else 0
            
            most_common_pattern = max(across_patterns.items(), key=lambda x: x[1])[0] if across_patterns else None
            features['most_common_comm_pattern'] = most_common_pattern
            features['comm_pattern_count'] = len(across_patterns)
        else:
            for key in ['across_count', 'across_total_width', 'across_max_width', 
                        'across_avg_width', 'most_common_comm_pattern', 'comm_pattern_count']:
                features[key] = 0
        
        # Remote access директивы
        features['remote_access_count'] = 0
        if 'remote_access' in directives and directives['remote_access'] is not None:
            features['remote_access_count'] = len(directives['remote_access'])
        
        # Reduction директивы
        if 'reduction' in directives and directives['reduction'] is not None:
            reductions = directives['reduction']
            features['reduction_count'] = len(reductions)
            
            total_reduction_size = 0
            reduction_operations = {}
            
            for red in reductions:
                total_reduction_size += red.get('size_bytes', 0)
                operation = red.get('operation', 'unknown')
                reduction_operations[operation] = reduction_operations.get(operation, 0) + 1
            
            features['total_reduction_size'] = total_reduction_size
            features['reduction_operation_types'] = len(reduction_operations)
            
            if reduction_operations:
                most_common_red_op = max(reduction_operations.items(), key=lambda x: x[1])[0]
                features['most_common_reduction_op'] = most_common_red_op
                features['most_common_reduction_freq'] = reduction_operations[most_common_red_op]
            else:
                features['most_common_reduction_op'] = None
                features['most_common_reduction_freq'] = 0
        else:
            for key in ['reduction_count', 'total_reduction_size', 'reduction_operation_types',
                        'most_common_reduction_op', 'most_common_reduction_freq']:
                features[key] = 0
        
        # Parallel директивы
        if 'parallel' in directives and directives['parallel'] is not None:
            parallel_dirs = directives['parallel']
            features['parallel_directives_count'] = len(parallel_dirs)
            
            intensities = []
            for p in parallel_dirs:
                ci = p.get('computational_intensity', None)
                if ci is not None and not np.isnan(ci):
                    intensities.append(ci)
                else:
                    intensities.append(0.0)
            
            features['min_computational_intensity'] = min(intensities) if intensities else 0
            features['max_computational_intensity'] = max(intensities) if intensities else 0
            features['avg_computational_intensity'] = np.mean(intensities) if intensities else 0
            features['std_computational_intensity'] = np.std(intensities) if intensities else 0
            
            total_iterations = 0
            total_loops = 0
            loop_depths = []
            
            for par in parallel_dirs:
                if 'iterations_count' in par and par['iterations_count']:
                    iter_count = par['iterations_count']
                    abs_iter_count = [abs(count) for count in iter_count]
                    total_iterations += sum(abs_iter_count)
                    loop_depths.append(len(abs_iter_count))
                
                total_loops += par.get('loops_count', 0)
            
            features['total_iteration_count'] = total_iterations
            features['total_loops_count'] = total_loops
            features['avg_iterations_per_loop'] = total_iterations / total_loops if total_loops > 0 else 0
            features['max_loop_depth'] = max(loop_depths) if loop_depths else 0
            features['avg_loop_depth'] = np.mean(loop_depths) if loop_depths else 0
            
            total_dependencies = 0
            total_shadow_renews_in_parallel = 0
            total_across_in_parallel = 0
            
            for par in parallel_dirs:
                reductions_in_par = par.get('reductions', []) or []
                acrosses_in_par = par.get('acrosses', []) or []
                shadow_renews_in_par = par.get('shadow_renews', []) or []
                
                total_dependencies += len(reductions_in_par) + len(acrosses_in_par)
                total_shadow_renews_in_parallel += len(shadow_renews_in_par)
                total_across_in_parallel += len(acrosses_in_par)
            
            features['total_dependencies'] = total_dependencies
            features['avg_dependencies_per_parallel'] = total_dependencies / len(parallel_dirs) if parallel_dirs else 0
            features['shadow_renews_in_parallel'] = total_shadow_renews_in_parallel
            features['across_in_parallel'] = total_across_in_parallel
        else:
            for key in ['parallel_directives_count', 'min_computational_intensity', 'max_computational_intensity',
                        'avg_computational_intensity', 'std_computational_intensity', 'total_iteration_count',
                        'total_loops_count', 'avg_iterations_per_loop', 'max_loop_depth', 'avg_loop_depth',
                        'total_dependencies', 'avg_dependencies_per_parallel', 'shadow_renews_in_parallel',
                        'across_in_parallel']:
                features[key] = 0
        
        # Вычисляемые признаки
        features['computation_to_communication_ratio'] = (
            features['avg_computational_intensity'] / (features['across_avg_width'] + 1e-6)
        )
        
        features['memory_to_computation_ratio'] = (
            features['total_array_size_bytes'] / (features['total_iteration_count'] + 1e-6)
        )
        
        features['parallel_fraction_efficiency_potential'] = (
            features['parallel_execution_fraction'] * features['avg_computational_intensity']
        )
        
        return features
    
    def extract_launch_configuration_features(self, grid: List[int], threads: int) -> Dict[str, float]:
        """
        Извлечение признаков конфигурации запуска
        
        Args:
            grid: Размерности сетки процессоров
            threads: Количество нитей
            
        Returns:
            Dict: Признаки конфигурации запуска
        """
        features = {}

        # Добавляем размерности сетки (до 4 измерений)
        for i in range(4):
            if len(grid) <= i:
                features[f'launch_grid_{i+1}'] = 1
                continue
            features[f'launch_grid_{i+1}'] = grid[i]

        # Вычисляем общее количество процессоров в сетке
        mul_grid = 1
        for g in grid:
            mul_grid *= g
                
        features['launch_threads'] = threads
        features['launch_total_processors'] = mul_grid * threads
        features['dvmh_num_threads'] = threads / self.dvmh_processors
        features['dvmh_grid_size'] = mul_grid
        features['processor_utilization'] = min(1.0, features['launch_total_processors'] / self.dvmh_processors)
        features['threads_per_dvmh_proc'] = threads / self.dvmh_processors
        features['parallelism_degree'] = mul_grid * threads
        features['is_power_of_two_threads'] = 1 if (threads & (threads - 1)) == 0 else 0
        features['is_power_of_two_grid'] = 1 if (mul_grid & (mul_grid - 1)) == 0 else 0
        features['potential_contention'] = 1 if threads > self.dvmh_processors else 0
        features['oversubscription_factor'] = threads / self.dvmh_processors if threads > self.dvmh_processors else 1.0
        
        return features
    
    def calculate_speedup(self, sequential_time: float, parallel_time: float) -> float:
        """
        Вычисление ускорения
        
        Args:
            sequential_time: Время последовательного выполнения
            parallel_time: Время параллельного выполнения
            
        Returns:
            float: Ускорение
        """
        if parallel_time == 0 or sequential_time == 0:
            return 0.0
        return sequential_time / parallel_time
    
    def create_feature_space(self, data: Dict, program_name: str) -> pd.DataFrame:
        """
        Создание признакового пространства для одной программы
        
        Args:
            data: Данные всех программ
            program_name: Имя программы для обработки
            
        Returns:
            pd.DataFrame: Признаковое пространство программы
        """
        if program_name not in data:
            raise ValueError(f"Program {program_name} not found in data")
        
        program_data = data[program_name]
        program_info = program_data.get('program_info', {})
        
        # Извлекаем статические признаки программы
        static_features = self.extract_static_program_features(program_info)
        
        # Получаем данные о запусках
        launches = program_info.get('launches', [])
        if not launches:
            raise ValueError(f"No launch data found for program {program_name}")
        
        rows = []

        # Определяем последовательное время как запуск на 1 потоке и 1 процессоре
        sequential_candidates = []
        for launch in launches:
            grid = launch.get('grid', [])
            threads = launch.get('threads', 0)
            if threads == 1 and grid and all(g == 1 for g in grid):
                sequential_candidates.append(launch.get('total_time', 0))

        if sequential_candidates:
            sequential_time = min(sequential_candidates)
        else:
            sequential_time = static_features['sequential_execution_time']
            self.logger.warning(
                "Не найден запуск с grid=1 и threads=1, используется sequential_execution_time"
            )
        
        # Обрабатываем каждый запуск
        for launch in launches:
            grid = launch['grid']
            threads = launch['threads']
            parallel_time = launch['total_time']
            
            # Создаем строку данных для этого запуска
            row = static_features.copy()
            row.update(self.extract_launch_configuration_features(grid, threads))
            
            # Добавляем целевые переменные
            row['target_speedup'] = self.calculate_speedup(sequential_time, parallel_time)
            row['parallel_execution_time'] = parallel_time
            
            # Добавляем метаданные
            row['program_name'] = program_name
            row['launch_config'] = f"grid={grid}_threads={threads}"
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_dataset_for_training(self, data: Dict) -> pd.DataFrame:
        """
        Создание полного датасета для обучения
        
        Args:
            data: Словарь с данными всех программ
            
        Returns:
            pd.DataFrame: Полный датасет для обучения
        """
        self.logger.info("Создание датасета для обучения")
        
        all_datasets = []
        
        for program_name in data.keys():
            try:
                df = self.create_feature_space(data, program_name)
                all_datasets.append(df)
                self.logger.debug(f"Обработана программа {program_name}: {len(df)} запусков")
            except Exception as e:
                self.logger.error(f"Error processing {program_name}: {e}")
        
        if not all_datasets:
            raise ValueError("No valid programs found in data")
        
        # Объединяем все датасеты
        final_df = pd.concat(all_datasets, ignore_index=True)
        
        # Добавляем дополнительные вычисляемые признаки
        final_df['normalized_parallel_time'] = final_df['parallel_execution_time'] / final_df['sequential_execution_time']
        
        # Валидация данных
        self._validate_dataset(final_df)
        
        self.logger.info(f"Датасет создан: {final_df.shape[0]} строк, {final_df.shape[1]} столбцов")
        return final_df
    
    def _validate_dataset(self, df: pd.DataFrame) -> None:
        """Валидация созданного датасета"""
        validation_config = self.config.get('feature_extraction', {}).get('validation', {})
        
        # Проверка на NaN значения
        if validation_config.get('check_nan_values', True):
            nan_counts = df.isnull().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"Найдены NaN значения в столбцах: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Проверка на бесконечные значения
        if validation_config.get('check_infinite_values', True):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_mask = np.isinf(df[numeric_cols]).any()
            if inf_mask.any():
                self.logger.warning(f"Найдены бесконечные значения в столбцах: {inf_mask[inf_mask].index.tolist()}")
        
        # Проверка дисперсии
        min_variance = validation_config.get('min_variance_threshold', 0.001)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_cols = []
        for col in numeric_cols:
            if col not in ['program_name', 'launch_config'] and df[col].var() < min_variance:
                low_variance_cols.append(col)
        
        if low_variance_cols:
            self.logger.warning(f"Столбцы с низкой дисперсией: {low_variance_cols}")
    
    def save_feature_info(self, df: pd.DataFrame, output_path: str) -> None:
        """Сохранение информации о признаках"""
        feature_info = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'feature_list': df.columns.tolist(),
            'feature_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'dataset_shape': df.shape,
            'programs_count': df['program_name'].nunique() if 'program_name' in df.columns else 0,
            'launches_count': len(df)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Информация о признаках сохранена в: {output_path}")