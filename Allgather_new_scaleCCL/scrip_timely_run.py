import threading
import subprocess
import os
import sys

# ================= 配置区域 =================
# 实验序号，修改此处即可改变输出文件名的后缀
EXP_ID = "gpu9" # alpha = 0.5, chunk_size = 4

# 需要执行的脚本列表
SCRIPTS = ["baseline.py", "fast_slow.py", "fast.py", "slow.py"]

# 输出保存的目录
OUTPUT_DIR = "logs"
# ===========================================

def run_script(script_name, exp_id):
    """
    执行单个脚本，捕获输出并写入文件
    """
    print(f"[开始执行] {script_name} (实验号: {exp_id})...")
    
    try:
        # 使用 sys.executable 确保使用当前环境的 Python 解释器
        # subprocess.PIPE 用于捕获标准输出
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将错误也重定向到输出
            text=True,
            encoding='utf-8'
        )

        # 获取输出内容
        stdout, _ = process.communicate()

        # 在当前控制台实时显示该脚本的运行结果
        print(f"\n{'='*20} {script_name} 输出结果 {'='*20}\n{stdout}")

        # 构造文件名: 脚本名(去掉.py)_exp_序号.txt
        base_name = os.path.splitext(script_name)[0]
        file_name = f"{base_name}_exp_{exp_id}.txt"
        file_path = os.path.join(OUTPUT_DIR, file_name)

        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(stdout)
        
        print(f"[完成] {script_name} 结果已保存至: {file_path}")

    except Exception as e:
        print(f"[错误] 执行 {script_name} 时出错: {str(e)}")

def main():
    # 创建保存目录（如果不存在）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    threads = []

    # 为每个脚本创建一个线程
    for script in SCRIPTS:
        t = threading.Thread(target=run_script, args=(script, EXP_ID))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    print("\n所有任务已执行完毕。")

if __name__ == "__main__":
    main()