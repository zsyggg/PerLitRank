# run_baselines.py
"""
运行基线方法的统一脚本 (移除连续性和Parquet逻辑)
- 完全依赖命令行参数。
- 假设所有数据集格式为 JSONL。
"""
import os
import argparse
import logging
import time
import subprocess
import sys
import importlib

# --- Setup Logger ---
logger = logging.getLogger("BaselinesRunner")
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
log_file = "baselines_runner.log" # Log to current directory
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
# --- PyTorch Check ---
try:
    import torch
except ImportError:
    logger.error("PyTorch is not installed. Please install it.")
    sys.exit(1)

def run_baseline(script_name, baseline_name, args):
    """运行单个基线脚本，传递必要的配置"""
    logger.info(f"--- Preparing to run baseline: {baseline_name} for dataset {args.dataset_name} ---")

    # 确定文件路径
    dataset_results_dir = os.path.join(args.results_dir, args.dataset_name)
    dataset_data_dir = os.path.join(args.data_dir, args.dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)

    # 文件路径 (假设 JSONL 格式)
    input_file = os.path.join(dataset_data_dir, "queries.jsonl")
    corpus_path = os.path.join(dataset_data_dir, "corpus.jsonl")
    corpus_embeddings_path = os.path.join(dataset_data_dir, "corpus_embeddings.npy")

    # 检索结果文件路径 (用于 CONQRR)
    # 假设 retrieved.jsonl 在主结果目录，而不是基线结果目录
    main_results_base = os.path.dirname(args.results_dir.rstrip('/')) # e.g., /workspace/PerMed
    main_dataset_results_dir = os.path.join(main_results_base, "results", args.dataset_name)
    retrieved_file = os.path.join(main_dataset_results_dir, "retrieved.jsonl")

    # 输出文件路径
    output_file = os.path.join(dataset_results_dir, f"{baseline_name}_results.jsonl")

    # 检查输入文件
    if not os.path.exists(input_file):
        logger.error(f"Input file not found for {baseline_name}: {input_file}. Skipping.")
        return False, None, 0
    if baseline_name in ["rpmn", "htps", "qer", "conqrr"]:
        if not os.path.exists(corpus_path):
             logger.error(f"Corpus file not found for {baseline_name}: {corpus_path}. Skipping.")
             return False, None, 0
        if not os.path.exists(corpus_embeddings_path):
             logger.error(f"Corpus embeddings not found for {baseline_name}: {corpus_embeddings_path}. Skipping.")
             return False, None, 0

    # 构建命令
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    command = [
        sys.executable,
        script_path,
        "--input_file", input_file,
        "--output_file", output_file,
        # 通用参数
        "--batch_size", str(args.batch_size),
        # Removed continuity_filter
        # Removed corpus_format (assumed jsonl in baseline scripts now)
        "--device", args.device,
    ]

    # 添加模型路径 (从 args 获取)
    encoder_path = args.encoder_model_path
    llm_path = args.llm_model_path

    if baseline_name in ["rpmn", "htps"]:
        if encoder_path:
             command.extend(["--model_path", encoder_path])
        else:
             logger.error(f"Encoder model path (--encoder_model_path) is required for {baseline_name} but not provided. Skipping.")
             return False, None, 0
        command.extend(["--corpus_path", corpus_path])
        command.extend(["--corpus_embeddings_path", corpus_embeddings_path])

    if baseline_name in ["qer", "conqrr"]:
        if llm_path:
            command.extend(["--model_path", llm_path])
        else:
            logger.error(f"LLM model path (--llm_model_path) is required for {baseline_name} but not provided. Skipping.")
            return False, None, 0
        if encoder_path:
             command.extend(["--encoder_model", encoder_path])
        else:
             logger.error(f"Encoder model path (--encoder_model_path) is required for {baseline_name} retrieval but not provided. Skipping.")
             return False, None, 0
        command.extend(["--corpus_path", corpus_path])
        command.extend(["--corpus_embeddings_path", corpus_embeddings_path])
        if args.no_retrieval:
            command.append("--no_retrieval")
        if baseline_name == "conqrr":
             if os.path.exists(retrieved_file):
                 command.extend(["--retrieved_file", retrieved_file])
             else:
                 logger.warning(f"Retrieved file not found for CONQRR: {retrieved_file}. CONQRR might not use retrieved docs.")

    logger.info(f"Executing command for {baseline_name}: {' '.join(command)}")

    # 执行命令
    start_time = time.time()
    success = False
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
        return_code = process.returncode

        if return_code == 0:
            logger.info(f"[{baseline_name}] executed successfully.")
            success = True
        else:
            logger.error(f"[{baseline_name}] failed with return code: {return_code}")
            error_log_path = os.path.join(dataset_results_dir, f"{baseline_name}_error.log")
            with open(error_log_path, 'w', encoding='utf-8') as err_f:
                 err_f.write(f"Command: {' '.join(command)}\n\n")
                 err_f.write("--- STDOUT ---\n")
                 err_f.write(process.stdout or "None")
                 err_f.write("\n\n--- STDERR ---\n")
                 err_f.write(process.stderr or "None")
            logger.error(f"See error details in: {error_log_path}")

    except FileNotFoundError:
        logger.error(f"[{baseline_name}] failed: Script '{script_path}' not found or Python interpreter issue.")
    except Exception as e:
        logger.error(f"[{baseline_name}] execution failed with exception: {e}", exc_info=True)

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- Finished baseline: {baseline_name} in {duration:.2f} seconds ---")
    return success, output_file, duration


def run_all_baselines(args):
    """运行所有指定的基线方法"""
    baseline_scripts = {
        "qer": "qer.py",
        "conqrr": "conqrr.py",
        "htps": "htps.py",
        "rpmn": "rpmn.py"
    }
    results_summary = {}

    # 检查必要的模型路径
    if not args.encoder_model_path:
        logger.error("Encoder model path (--encoder_model_path) is required but not provided. Exiting.")
        sys.exit(1)
    if not args.llm_model_path and ("qer" in args.baselines or "conqrr" in args.baselines):
        logger.error("LLM model path (--llm_model_path) is required for QER/CONQRR but not provided. Exiting.")
        sys.exit(1)

    for baseline_name in args.baselines:
        script = baseline_scripts.get(baseline_name)
        if script:
            success, output_f, duration = run_baseline(script, baseline_name, args)
            results_summary[baseline_name] = {
                "success": success,
                "output_file": output_f,
                "time": duration
            }
        else:
            logger.warning(f"Baseline '{baseline_name}' is not recognized. Skipping.")

    # 输出结果摘要
    logger.info("\n====== Baseline Execution Summary ({}) ======".format(args.dataset_name))
    for name, result in results_summary.items():
        status = "Success" if result["success"] else "Failed"
        time_str = f"{result['time']:.2f}s" if result['time'] is not None else "N/A"
        output_str = result['output_file'] if result['output_file'] else "N/A"
        logger.info(f"{name.upper():<10}: {status:<8} - Output: {output_str} (Time: {time_str})")

    return results_summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Run baseline methods")
    # 数据集和路径参数
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="Dataset name (e.g., LitSearch, MedCorpus, CORAL)")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="Base data directory")
    parser.add_argument("--results_dir", type=str, default="./baselines/results",
                       help="Base results directory for baselines")
    # Removed corpus_format argument

    # 模型路径参数 (现在是必须的)
    parser.add_argument("--llm_model_path", type=str, required=True,
                       help="LLM model path (Required for QER, CONQRR)")
    parser.add_argument("--encoder_model_path", type=str, required=True,
                       help="Encoder model path (Required for RPMN, HTPS, QER, CONQRR)")

    # 执行控制参数
    parser.add_argument("--baselines", nargs='+', type=str,
                       default=["qer", "conqrr", "htps", "rpmn"],
                       help="List of baselines to run")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (e.g., 'cuda:0', 'cpu'). Defaults to cuda:0 if available, else cpu.")
    parser.add_argument("--no_retrieval", action="store_true",
                       help="Skip retrieval step (for QER/CONQRR if applicable)")
    # Removed continuity_filter argument

    args = parser.parse_args()

    # 确定设备
    if args.device:
        if not (args.device == "cpu" or (args.device.startswith("cuda:") and args.device.split(':')[1].isdigit())):
             logger.error(f"Invalid device format: '{args.device}'. Use 'cpu' or 'cuda:N'. Exiting.")
             sys.exit(1)
    else:
        # Check PyTorch availability before setting default device
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "cuda:0" # Default GPU 0
            else:
                args.device = "cpu"
        except ImportError:
             logger.warning("PyTorch not found. Defaulting device to 'cpu'.")
             args.device = "cpu"


    logger.info(f"Starting baseline runs for dataset: {args.dataset_name}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Results directory: {args.results_dir}") # 基线结果目录
    logger.info(f"Running baselines: {args.baselines}")
    # logger.info(f"Continuity filter: {args.continuity_filter}") # Removed
    logger.info(f"Device: {args.device}")
    logger.info(f"Encoder Path: {args.encoder_model_path}")
    logger.info(f"LLM Path: {args.llm_model_path}")

    # 运行所有选定的基线
    run_all_baselines(args)

    logger.info(f"Finished baseline runs for dataset: {args.dataset_name}")

if __name__ == "__main__":
    main()
