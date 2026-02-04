import os
import shutil
import logging
import datetime
import glob
import yaml
from typing import List, Dict, Optional
from .utils import run_command

logger = logging.getLogger(__name__)

class DVMHDataCollector:
    """Класс для сбора данных покрытия DVMH программ"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Инициализация коллектора данных
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.logger = logging.getLogger(f"dvmh_predictor.{__name__}")
        self.config = self._load_config(config_path)

        collection_cfg = self.config.get('data_collection', {})
        # dimensions могут отсутствовать в новом конфиге, тогда определяем их динамически
        self.dimensions = collection_cfg.get('dimensions') or []
        self.file_extensions = collection_cfg.get('file_extensions', ['.f', '.f90'])
        self.compiler_flags = collection_cfg.get('compiler_flags', '-O3 -g -fprofile-arcs -ftest-coverage')
        self.temp_patterns = collection_cfg.get('temp_file_patterns', ['*.gcda', '*.gcno', '*.gcov', '*.dep', '*.proj', '*.mod'])
        self.preserved_files = collection_cfg.get('preserved_files', ['info.json', '*_output.log', '*.gcov'])
        self.sapfor_path = collection_cfg.get('sapfor_executable', '../SAPFOR/_bin/Release/Sapfor_F.exe')
        
        self.logger.info("DVMHDataCollector инициализирован")
    
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.error(f"Конфигурационный файл не найден: {config_path}")
            # Возвращаем конфигурацию по умолчанию
            return {
                'data_collection': {
                    'dimensions': ['1d', '2d', '3d', '4d'],
                    'file_extensions': ['.f', '.f90'],
                    'compiler_flags': '-O3 -g -fprofile-arcs -ftest-coverage',
                    'temp_file_patterns': ['*.gcda', '*.gcno', '*.gcov', '*.dep', '*.proj', '*.mod'],
                    'preserved_files': ['info.json', '*_output.log', '*.gcov'],
                    'sapfor_executable': '../SAPFOR/_bin/Release/Sapfor_F.exe'
                }
            }
    
    def process_all_programs(self, source_dir: str, results_dir: str) -> bool:
        """
        Обработка всех программ из исходной директории
        
        Args:
            source_dir: Путь к директории с исходными программами
            results_dir: Путь к директории для сохранения результатов
            
        Returns:
            bool: True если обработка прошла успешно
        """
        self.logger.info(f"Начало обработки программ из {source_dir}")
        
        # Проверяем существование исходной директории
        if not os.path.exists(source_dir):
            self.logger.error(f"Исходная директория не существует: {source_dir}")
            return False
        
        # Создаем директорию результатов
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            self.logger.info(f"Создана директория результатов: {results_dir}")
        
        success_count = 0
        total_count = 0
        
        # Определяем список размерностей (или просто подпапок) для обработки
        dimensions = self.dimensions or self._discover_dimensions(source_dir)

        # Обрабатываем каждую размерность/подпапку
        for dim in dimensions:
            dim_success, dim_total = self._process_dimension(source_dir, results_dir, dim)
            success_count += dim_success
            total_count += dim_total
        
        # Очищаем временные файлы
        self._cleanup_temp_files()
        
        self.logger.info(f"Обработка завершена. Успешно: {success_count}/{total_count}")
        return success_count > 0
    
    def _process_dimension(self, source_dir: str, results_dir: str, dim: str) -> tuple:
        """
        Обработка программ конкретной размерности
        
        Args:
            source_dir: Исходная директория
            results_dir: Директория результатов
            dim: Размерность (1d, 2d, 3d, 4d)
            
        Returns:
            tuple: (количество успешных, общее количество)
        """
        self.logger.info(f"Обработка размерности: {dim}")
        
        # Создаем папку для размерности
        results_dim_path = os.path.join(results_dir, dim)
        if not os.path.exists(results_dim_path):
            os.makedirs(results_dim_path)
        
        # Получаем список файлов в исходной директории
        dim_dir = os.path.join(source_dir, dim)
        if not os.path.exists(dim_dir):
            self.logger.warning(f"Директория размерности не найдена: {dim_dir}")
            return 0, 0
        
        files = os.listdir(dim_dir)
        program_files = [f for f in files if any(f.endswith(ext) for ext in self.file_extensions)]
        
        success_count = 0
        
        for file in program_files:
            if self._process_single_file(dim_dir, results_dim_path, file):
                success_count += 1
        
        self.logger.info(f"Размерность {dim}: обработано {success_count}/{len(program_files)} файлов")
        return success_count, len(program_files)
    
    def _process_single_file(self, source_dim_dir: str, results_dim_dir: str, filename: str) -> bool:
        """
        Обработка одного файла программы
        
        Args:
            source_dim_dir: Директория с исходными файлами размерности
            results_dim_dir: Директория результатов для размерности
            filename: Имя файла для обработки
            
        Returns:
            bool: True если обработка прошла успешно
        """
        file_name = filename.split('.')[0]
        self.logger.info(f"Обработка файла: {filename}")
        
        # Создание директории для файла
        file_dir_path = os.path.join(results_dim_dir, file_name)
        if not os.path.exists(file_dir_path):
            os.makedirs(file_dir_path)
        
        # Создаем файл для вывода программы
        output_log_file = os.path.join(file_dir_path, f"{file_name}_output.log")
        
        # Инициализируем файл вывода
        self._initialize_log_file(output_log_file, filename, 
                                 os.path.basename(results_dim_dir))
        
        # Копируем исходный файл (целевая копия с расширением .f)
        source_file = os.path.join(source_dim_dir, filename)
        base_name, _ = os.path.splitext(filename)
        dest_file = os.path.join(file_dir_path, f"{base_name}.f")
        if not self._copy_source_file(source_file, dest_file, output_log_file):
            return False
        
        # Выполняем основную обработку
        try:
            return self._execute_processing_pipeline(file_dir_path, dest_file, 
                                                   file_name, filename, output_log_file)
        except Exception as e:
            error_msg = f"Ошибка при обработке {filename}: {str(e)}"
            self.logger.error(error_msg)
            self._log_error_to_file(output_log_file, filename, error_msg)
            return False
    
    def _initialize_log_file(self, output_log_file: str, filename: str, dimension: str):
        """Инициализация лог-файла для обработки"""
        with open(output_log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Processing file: {filename} ===\n")
            f.write(f"Date and time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dimension: {dimension}\n")
            f.write("=" * 50 + "\n\n")
    
    def _copy_source_file(self, source_file: str, dest_file: str, output_log_file: str) -> bool:
        """Копирование исходного файла"""
        if not os.path.exists(dest_file):
            try:
                shutil.copy2(source_file, dest_file)
                message = f"Copied {source_file} to {dest_file}"
                self.logger.debug(message)
                with open(output_log_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
                return True
            except Exception as e:
                error_msg = f"Error copying file {source_file} to {dest_file}: {str(e)}"
                self.logger.error(error_msg)
                with open(output_log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                return False
        return True
    
    def _execute_processing_pipeline(self, file_dir_path: str, dest_file: str, 
                                   file_name: str, filename: str, output_log_file: str) -> bool:
        """Выполнение основного пайплайна обработки"""
        
        # 1. Компиляция с профилированием
        source_basename = os.path.basename(dest_file)
        executable_name = file_name
        compile_cmd = f'gfortran {self.compiler_flags} "{source_basename}" -o "{executable_name}"'
        
        result = run_command(compile_cmd, check=False, output_file=output_log_file, cwd=file_dir_path)
        if result is False:
            self.logger.error(f"Ошибка компиляции для {filename}")
            return False
        
        # 2. Запуск программы
        run_prog_cmd = f'"./{executable_name}"'
        result = run_command(run_prog_cmd, check=False, output_file=output_log_file, cwd=file_dir_path)
        if result is False:
            self.logger.warning(f"Ошибка выполнения программы {filename}")
            
            
        # 3. Генерация отчета покрытия
        gcov_cmd = f'gcov -b "{source_basename}"'
        run_command(gcov_cmd, check=False, output_file=output_log_file, cwd=file_dir_path)
        
        # 4. Анализ SAPFOR
        if not self._run_sapfor_analysis(source_basename, file_dir_path, output_log_file):
            self.logger.warning(f"Ошибка анализа SAPFOR для {filename}")
        
        # 6. Финализация лога
        with open(output_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Processing of {filename} completed successfully at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return True
    
    def _run_sapfor_analysis(self, source_basename: str, file_dir_path: str, output_log_file: str) -> bool:
        """Запуск анализа SAPFOR"""
        try:
            # Парсинг файла
            sapfor_parse_cmd = f'"{self.sapfor_path}" -parse -spf "{source_basename}"'
            run_command(sapfor_parse_cmd, check=False, output_file=output_log_file, cwd=file_dir_path)
            
            # Получение статистик
            sapfor_stats_cmd = f'"{self.sapfor_path}" -passN GET_STATS_FOR_PREDICTOR -keepDVM'
            run_command(sapfor_stats_cmd, check=False, output_file=output_log_file, cwd=file_dir_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка в анализе SAPFOR: {str(e)}")
            return False
    
    def copy_and_remove(self, source: str, dest: str, output_file: Optional[str] = None) -> bool:
        """Копирует файл и удаляет оригинал"""
        try:
            if os.path.exists(source):
                shutil.copy2(source, dest)
                os.remove(source)
                message = f"Copied and removed {source} to {dest}"
                self.logger.debug(message)
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(message + "\n")
                return True
            else:
                error_msg = f"Исходный файл {source} не существует, копирование невозможно"
                self.logger.warning(error_msg)
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(error_msg + "\n")
                return False
        except Exception as e:
            error_msg = f"Error copying/removing {source} to {dest}: {str(e)}"
            self.logger.error(error_msg)
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
            return False
    
    def cleanup_results_directories(self, results_dir: str = './data/raw'):
        """Очистка директорий результатов, оставляем только важные файлы"""
        self.logger.info("Очистка директорий результатов...")
        
        if not os.path.exists(results_dir):
            self.logger.info(f"Директория {results_dir} не существует, очистка не требуется")
            return
        
        cleaned_count = 0
        
        # Перебираем все поддиректории
        for root, dirs, files in os.walk(results_dir):
            # Пропускаем корневую директорию
            if root == results_dir:
                continue
            
            # Проверяем каждый файл
            for file in files:
                file_path = os.path.join(root, file)
                
                # Проверяем, нужно ли сохранить файл
                should_preserve = any(
                    file == preserved or 
                    (preserved.startswith('*') and file.endswith(preserved[1:])) or
                    (preserved.endswith('*') and file.startswith(preserved[:-1]))
                    for preserved in self.preserved_files
                )
                
                if not should_preserve:
                    try:
                        os.remove(file_path)
                        self.logger.debug(f"Удален файл: {file_path}")
                        cleaned_count += 1
                    except Exception as e:
                        error_msg = f"Не удалось удалить файл {file_path}: {str(e)}"
                        self.logger.error(error_msg)
        
        self.logger.info(f"Очистка завершена. Удалено файлов: {cleaned_count}")
    
    def find_directories_without_info_json(self, results_dir: str = './data/raw') -> List[str]:
        """Поиск директорий без файла info.json"""
        self.logger.info("Поиск директорий без файла info.json...")
        
        if not os.path.exists(results_dir):
            self.logger.warning(f"Директория {results_dir} не существует")
            return []
        
        directories_without_info = []
        
        # Перебираем все поддиректории в results
        for root, dirs, files in os.walk(results_dir):
            # Пропускаем корневую директорию и директории верхнего уровня (1d, 2d, 3d, 4d)
            if root == results_dir or os.path.dirname(root) == results_dir:
                continue
            
            # Проверяем наличие info.json
            if 'info.json' not in files:
                directories_without_info.append(root)
                self.logger.debug(f"Директория без info.json: {root}")
        
        self.logger.info(f"Всего найдено директорий без info.json: {len(directories_without_info)}")
        return directories_without_info
    
    def collect_problem_files(self, source_dir: str, results_dir: str = './data/raw', 
                            problems_dir: str = './problems'):
        """Сбор проблемных файлов в отдельную папку"""
        self.logger.info("Сбор проблемных файлов...")
        
        # Создаем директорию problems, если её не существует
        if not os.path.exists(problems_dir):
            os.makedirs(problems_dir)
            self.logger.info(f"Создана директория {problems_dir}")
        
        # Получаем список директорий без info.json
        directories_without_info = self.find_directories_without_info_json(results_dir)
        
        if not directories_without_info:
            self.logger.info("Не найдено проблемных файлов")
            return
        
        for problem_dir in directories_without_info:
            self._collect_single_problem_file(problem_dir, source_dir, problems_dir)
        
        self.logger.info(f"Завершен сбор проблемных файлов. Всего обработано: {len(directories_without_info)}")
    
    def _collect_single_problem_file(self, problem_dir: str, source_dir: str, problems_dir: str):
        """Сбор одного проблемного файла"""
        # Извлекаем измерение и имя файла из пути
        parts = problem_dir.replace('\\', '/').split('/')
        if len(parts) < 3:
            self.logger.warning(f"Некорректный путь директории: {problem_dir}")
            return
        
        dimension = parts[-2]  # Измерение (1d, 2d и т.д.)
        file_name = parts[-1]  # Имя файла без расширения
        
        # Пути к файлам
        source_file_path = os.path.join(source_dir, dimension, f"{file_name}.f")
        log_file_path = os.path.join(problem_dir, f"{file_name}_output.log")
        gcov_file_path = os.path.join(problem_dir, f"{file_name}.f.gcov")
        
        # Создаем поддиректорию в problems для этого файла
        problem_file_dir = os.path.join(problems_dir, file_name)
        if not os.path.exists(problem_file_dir):
            os.makedirs(problem_file_dir)
        
        # Копируем файлы
        self._copy_problem_file(source_file_path, problem_file_dir, f"{file_name}.f", "исходный файл")
        self._copy_problem_file(log_file_path, problem_file_dir, f"{file_name}_output.log", "файл лога")
        self._copy_problem_file(gcov_file_path, problem_file_dir, f"{file_name}.gcov", "файл .gcov")
    
    def _copy_problem_file(self, source_path: str, dest_dir: str, dest_filename: str, file_type: str):
        """Копирование проблемного файла"""
        dest_path = os.path.join(dest_dir, dest_filename)
        
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Скопирован {file_type}: {source_path} -> {dest_path}")
            except Exception as e:
                error_msg = f"Ошибка при копировании {file_type} {source_path}: {str(e)}"
                self.logger.error(error_msg)
        else:
            # Создаем пустой файл с сообщением для логов
            if "лога" in file_type:
                try:
                    with open(dest_path, 'w', encoding='utf-8') as f:
                        f.write(f"{file_type.capitalize()} не найден в исходной директории: {source_path}\n")
                    self.logger.debug(f"Создан пустой {file_type} с уведомлением: {dest_path}")
                except Exception as e:
                    error_msg = f"Ошибка при создании пустого {file_type} {dest_path}: {str(e)}"
                    self.logger.error(error_msg)
            else:
                self.logger.warning(f"{file_type.capitalize()} не найден: {source_path}")
    
    def _cleanup_temp_files(self):
        """Очистка временных файлов"""
        self.logger.info("Очистка временных файлов...")
        
        cleaned_count = 0
        for pattern in self.temp_patterns:
            try:
                files = glob.glob(pattern)
                for file in files:
                    try:
                        os.remove(file)
                        self.logger.debug(f"Cleaned up temporary file: {file}")
                        cleaned_count += 1
                    except Exception as e:
                        error_msg = f"Failed to remove temp file {file}: {str(e)}"
                        self.logger.error(error_msg)
            except Exception as e:
                error_msg = f"Error during cleanup of pattern {pattern}: {str(e)}"
                self.logger.error(error_msg)
        
        self.logger.info(f"Очистка временных файлов завершена. Удалено: {cleaned_count}")
    
    def _log_error_to_file(self, output_log_file: str, filename: str, error_msg: str):
        """Запись ошибки в лог-файл"""
        with open(output_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"ERROR: Processing of {filename} failed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error message: {error_msg}\n")

    def _discover_dimensions(self, source_dir: str):
        """Автоопределение подпапок с исходниками, если список размерностей не задан"""
        try:
            return [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        except FileNotFoundError:
            self.logger.error(f"Исходная директория не существует: {source_dir}")
            return []