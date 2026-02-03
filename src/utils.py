import subprocess
import os
import shutil
import logging
import datetime
from typing import Union, Optional, Tuple

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> str:
    """
    Настройка логирования для всего приложения
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_dir: Директория для сохранения логов
        
    Returns:
        str: Путь к созданному лог-файлу
    """
    # Создаем директорию для логов
    os.makedirs(log_dir, exist_ok=True)
    
    # Генерируем имя файла с текущей датой
    log_filename = os.path.join(log_dir, f"app_{datetime.datetime.now().strftime('%Y%m%d')}.log")
    
    # Настраиваем логирование
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Логируем начало работы
    root_logger = logging.getLogger()
    root_logger.info("Логирование настроено успешно")
    root_logger.info(f"Лог-файл: {log_filename}")
    
    return log_filename

def run_command(command: str, check: bool = False, output_file: Optional[str] = None, cwd: Optional[str] = None) -> Union[subprocess.CompletedProcess, Tuple[bool, Exception]]:
    """
    Выполнение команды с логированием и опциональной записью в файл
    
    Args:
        command: Команда для выполнения
        check: Если True, выбрасывает исключение при ошибке
        output_file: Файл для записи вывода команды
        
    Returns:
        subprocess.CompletedProcess или (False, Exception) в случае ошибки
    """
    logger.info(f"Executing: {command}")
    
    try:
        result = subprocess.run(
            command,
            check=check,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=cwd
        )
        
        # Логируем результат
        if result.stdout:
            logger.debug(f"Command output: {result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"Command stderr: {result.stderr.strip()}")
        
        # Записываем в файл, если указан
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"Command: {command}\n")
                f.write(f"Return code: {result.returncode}\n")
                if result.stdout:
                    f.write(f"Output:\n{result.stdout}\n")
                if result.stderr:
                    f.write(f"Error output:\n{result.stderr}\n")
                f.write("-" * 50 + "\n")
                
        return result
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with return code {e.returncode}: {command}"
        if e.stderr:
            error_msg += f"\nError: {e.stderr}"
            
        logger.error(error_msg)
        
        # Записываем ошибку в файл
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"FAILED Command: {command}\n")
                f.write(f"Return code: {e.returncode}\n")
                if e.stderr:
                    f.write(f"Error: {e.stderr}\n")
                f.write("-" * 50 + "\n")
                
        return False, e
    except Exception as e:
        error_msg = f"Unexpected error executing command: {command}\nError: {str(e)}"
        logger.error(error_msg)
        
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"FAILED Command: {command}\n")
                f.write(f"Unexpected error: {str(e)}\n")
                f.write("-" * 50 + "\n")
                
        return False, e
