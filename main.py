import logging
import logging.config
import os
import argparse
import sys
from datetime import datetime
import pandas as pd


from src.model_trainer import DVMHModelTrainer        
from src.utils import setup_logging
from src.data_collector import DVMHDataCollector
from src.stats_aggregator import DVMHStatsAggregator
from src.feature_extractor import DVMHFeatureSpaceCreator


def setup_argument_parser():
    """Настройка парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="DVMH Performance Predictor - система предсказания эффективности DVMH программ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Режимы работы:
  collect     - сбор данных покрытия из исходных программ
  aggregate   - агрегация данных покрытия и результатов запусков
  features    - создание признакового пространства
  train       - обучение модели предсказания
  predict     - предсказание эффективности для новых данных
  pipeline    - полный пайплайн: aggregate + features
  
Примеры использования:
  python main.py collect --source ./sources --results ./data/raw
  python main.py aggregate --config config/data_config.yaml
  python main.py features --input aggregated_data.json --output dataset.csv
  python main.py pipeline --config config/data_config.yaml
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['collect', 'aggregate', 'features', 'train', 'predict', 'pipeline'],
        help='Режим работы приложения'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/data_config.yaml',
        help='Путь к конфигурационному файлу (по умолчанию: config/data_config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод (DEBUG уровень логирования)'
    )
    
    # Аргументы для режима collect
    collect_group = parser.add_argument_group('collect', 'Аргументы для сбора данных')
    collect_group.add_argument(
        '--source',
        default='./sources',
        help='Директория с исходными программами (по умолчанию: ./sources)'
    )
    collect_group.add_argument(
        '--results',
        default='./data/raw',
        help='Директория для сохранения результатов (по умолчанию: ./data/raw)'
    )
    
    # Аргументы для режимов aggregate и features
    data_group = parser.add_argument_group('data processing', 'Аргументы для обработки данных')
    data_group.add_argument(
        '--input', '-i',
        default='aggregated_data.json',
        help='Входной файл с агрегированными данными (по умолчанию: aggregated_data.json)'
    )
    data_group.add_argument(
        '--output', '-o',
        default='feature_dataset.csv',
        help='Выходной файл с признаковым пространством (по умолчанию: feature_dataset.csv)'
    )
    data_group.add_argument(
        '--processors', '-p',
        type=int,
        default=1,
        help='Количество DVMH процессоров (по умолчанию: 1)'
    )
    
    # Аргументы для режимов train и predict
    ml_group = parser.add_argument_group('machine learning', 'Аргументы для ML')
    ml_group.add_argument(
        '--dataset',
        help='Путь к датасету для обучения'
    )
    ml_group.add_argument(
        '--model',
        help='Путь к модели'
    )
    
    return parser


def mode_collect(args):
    """Режим сбора данных покрытия"""
    logger = logging.getLogger(__name__)
    logger.info("=== РЕЖИМ: Сбор данных покрытия ===")
    
    collector = DVMHDataCollector(args.config)
    
    # Проверяем существование исходной директории
    if not os.path.exists(args.source):
        logger.error(f"Исходная директория не найдена: {args.source}")
        return 1
    
    # Запускаем сбор данных
    success = collector.process_all_programs(args.source, args.results)
    
    if success:
        logger.info("Сбор данных завершен успешно")
        
        # Очистка результатов (удаление временных файлов)
        collector.cleanup_results_directories(args.results)
        
        # Сбор проблемных файлов
        collector.collect_problem_files(args.source, args.results)
        
        return 0
    else:
        logger.error("Ошибка при сборе данных")
        return 1


def mode_aggregate(args):
    """Режим агрегации данных"""
    logger = logging.getLogger(__name__)
    logger.info("=== РЕЖИМ: Агрегация данных ===")
    
    aggregator = DVMHStatsAggregator(args.config)
    
    # Агрегируем данные
    aggregated_data = aggregator.aggregate_all_statistics()
    
    if not aggregated_data:
        logger.error("Не удалось агрегировать данные")
        return 1
    
    # Сохраняем агрегированные данные
    if not aggregator.save_aggregated_data(aggregated_data, args.input):
        logger.error("Не удалось сохранить агрегированные данные")
        return 1
    
    # Генерируем отчет
    report = aggregator.generate_statistics_report(aggregated_data)
    logger.info("Отчет по агрегированным данным:")
    for key, value in report.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Агрегация данных завершена успешно")
    return 0


def mode_features(args):
    """Режим создания признакового пространства"""
    logger = logging.getLogger(__name__)
    logger.info("=== РЕЖИМ: Создание признакового пространства ===")
    
    aggregator = DVMHStatsAggregator(args.config)
    
    # Загружаем агрегированные данные
    aggregated_data = aggregator.load_aggregated_data(args.input)
    if aggregated_data is None:
        logger.error(f"Не удалось загрузить агрегированные данные из {args.input}")
        return 1
    
    # Создаем экстрактор признаков
    feature_extractor = DVMHFeatureSpaceCreator(dvmh_processors=args.processors, config_path=args.config)
    
    # Создаем датасет
    logger.info("Создание признакового пространства...")
    try:
        dataset = feature_extractor.create_dataset_for_training(aggregated_data)
        
        # Сохраняем датасет
        output_path = os.path.join(aggregator.output_path, args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        dataset.to_csv(output_path, index=False)
        logger.info(f"Признаковое пространство сохранено в: {output_path}")
        logger.info(f"Размер датасета: {dataset.shape}")
        logger.info(f"Количество признаков: {len([col for col in dataset.columns if col not in ['program_name', 'launch_config']])}")
        
        # Сохраняем информацию о признаках
        features_info_path = os.path.join(aggregator.output_path, "features_info.json")
        feature_extractor.save_feature_info(dataset, features_info_path)
        
        # Выводим основную статистику по датасету
        logger.info("Статистика по датасету:")
        logger.info(f"  Количество программ: {dataset['program_name'].nunique()}")
        logger.info(f"  Количество запусков: {len(dataset)}")
        logger.info(f"  Средний speedup: {dataset['target_speedup'].mean():.2f}")
        logger.info(f"  Максимальный speedup: {dataset['target_speedup'].max():.2f}")
        
        logger.info("Создание признакового пространства завершено успешно")
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка при создании признакового пространства: {str(e)}")
        if args.verbose:
            logger.exception("Детали ошибки:")
        return 1


def mode_train(args):
    """Режим обучения модели"""
    logger = logging.getLogger(__name__)
    logger.info("=== РЕЖИМ: Обучение модели ===")
    
    # Проверяем наличие датасета
    if not args.dataset:
        logger.error("Необходимо указать путь к датасету с помощью --dataset")
        return 1
    
    if not os.path.exists(args.dataset):
        logger.error(f"Датасет не найден: {args.dataset}")
        return 1
    
    try:
        # Загружаем датасет
        logger.info(f"Загрузка датасета: {args.dataset}")
        df = pd.read_csv(args.dataset, low_memory=False)
        logger.info(f"Загружен датасет: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        # Инициализируем тренер
        trainer = DVMHModelTrainer(args.config)
        
        # Запускаем обучение
        results = trainer.run_full_training(
            df=df,
            target_column='target_speedup',
            test_programs_count=3,
            epochs=100,
            batch_size=128
        )
        
        # Выводим результаты
        logger.info("Результаты обучения:")
        logger.info(f"MLP R²: {results['mlp_metrics']['r2']:.4f}")
        logger.info(f"Attention R²: {results['attention_metrics']['r2']:.4f}")
        
        logger.info("Обучение модели завершено успешно")
        return 0
        
    except ImportError:
        logger.error("Для обучения моделей необходимо установить PyTorch")
        logger.info("Выполните: pip install torch torchvision")
        return 1
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        return 1


def mode_predict(args):
    """Режим предсказания"""
    logger = logging.getLogger(__name__)
    logger.info("=== РЕЖИМ: Предсказание ===")
    
    try:
        from src.predictor import DVMHPerformancePredictor
        
        # Инициализируем предиктор
        predictor = DVMHPerformancePredictor(args.config)
        
        # Проверяем, есть ли обученные модели
        models_dir = predictor.models_dir
        if not os.path.exists(models_dir):
            logger.error(f"Директория с моделями не найдена: {models_dir}")
            logger.info("Сначала обучите модель с помощью команды 'train'")
            return 1
        
        # Ищем доступные модели
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        if not model_files:
            logger.error("Не найдено обученных моделей")
            logger.info("Сначала обучите модель с помощью команды 'train'")
            return 1
        
        # Загружаем первую доступную модель
        model_name = model_files[0].replace('.pt', '')
        model_type = 'attention' if 'attention' in model_name else 'mlp'
        
        if not predictor.load_model(model_name, model_type):
            logger.error("Не удалось загрузить модель")
            return 1
        
        # Получаем информацию о модели
        model_info = predictor.get_model_info(model_name)
        logger.info(f"Загружена модель: {model_info['model_name']} ({model_info['model_type']})")
        logger.info(f"Метрики модели: R² = {model_info['metrics'].get('r2', 'N/A')}")
        
        logger.info("Предсказание готово к использованию")
        logger.info("Для подробного предсказания используйте scripts/predict.py")
        
        return 0
        
    except ImportError:
        logger.error("Для предсказания необходимо установить PyTorch")
        logger.info("Выполните: pip install torch torchvision")
        return 1
    except Exception as e:
        logger.error(f"Ошибка в режиме предсказания: {str(e)}")
        return 1


def mode_pipeline(args):
    """Режим полного пайплайна: агрегация + создание признаков"""
    logger = logging.getLogger(__name__)
    logger.info("=== РЕЖИМ: Полный пайплайн ===")
    
    # Шаг 1: Агрегация
    logger.info("Шаг 1/2: Агрегация данных...")
    result = mode_aggregate(args)
    if result != 0:
        return result
    
    # Шаг 2: Создание признаков
    logger.info("Шаг 2/2: Создание признакового пространства...")
    result = mode_features(args)
    if result != 0:
        return result
    
    logger.info("Полный пайплайн завершен успешно")
    return 0


def main():
    """Главная функция приложения"""
    
    # Парсинг аргументов
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Настройка логирования
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("DVMH Performance Predictor запущен")
    logger.info(f"Режим: {args.mode}")
    logger.info(f"Конфигурация: {args.config}")
    logger.info(f"Лог-файл: {log_file}")
    logger.info("="*60)
    
    # Проверяем существование конфигурационного файла
    if not os.path.exists(args.config):
        logger.warning(f"Конфигурационный файл не найден: {args.config}")
        logger.info("Будут использованы настройки по умолчанию")
    
    # Выбор режима работы
    try:
        if args.mode == 'collect':
            exit_code = mode_collect(args)
        elif args.mode == 'aggregate':
            exit_code = mode_aggregate(args)
        elif args.mode == 'features':
            exit_code = mode_features(args)
        elif args.mode == 'train':
            exit_code = mode_train(args)
        elif args.mode == 'predict':
            exit_code = mode_predict(args)
        elif args.mode == 'pipeline':
            exit_code = mode_pipeline(args)
        else:
            logger.error(f"Неизвестный режим: {args.mode}")
            exit_code = 1
            
    except KeyboardInterrupt:
        logger.info("Выполнение прервано пользователем")
        exit_code = 130
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        if args.verbose:
            logger.exception("Детали ошибки:")
        exit_code = 1
    
    # Завершение
    if exit_code == 0:
        logger.info("Выполнение завершено успешно")
    else:
        logger.error(f"Выполнение завершено с ошибкой (код: {exit_code})")
    
    logger.info("="*60)
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
