import psutil
import smtplib
import time
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# ======================
# ✉️ 邮箱配置（使用 Gmail SMTP）
# ======================
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_SENDER = 'jiahengxiong1@gmail.com'
EMAIL_PASSWORD = 'xrmclemmkikqhpfs'   # 请确保这里填写的是你的 Gmail 应用专用密码（无空格）
EMAIL_RECEIVER = '10886580@polimi.it'

# ======================
# 📋 获取 python/python3 进程
# ======================
def get_python_processes():
    result = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = p.info.get('cmdline')
            if isinstance(cmdline, list) and any('python' in part for part in cmdline):
                result.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return result

# ======================
# 📧 发送邮件
# ======================
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = Header(subject, 'utf-8')
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # 启用 TLS 加密
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("[✓] 邮件已发送")
    except Exception as e:
        print("[✗] 邮件发送失败:", e)

# ======================
# 辅助函数：判断进程退出原因
# ======================
def get_exit_reason(pid):
    """
    通过检查 dmesg 输出来判断进程是否被系统杀掉（例如 OOM）。
    如果 dmesg 输出中同时包含 "killed process <pid>" 和 "out of memory"/"oom killer"
    则认为是被系统杀掉；否则判断为正常退出（例如手动 kill）。
    注意：这需要系统允许非 root 用户访问 dmesg 信息。
    """
    try:
        output = subprocess.check_output(['dmesg', '-T'], stderr=subprocess.STDOUT).decode('utf-8')
    except Exception:
        output = ""
    output_lower = output.lower()
    # 如果输出同时包含 "killed process <pid>" 与 "out of memory" 或 "oom killer"
    if f"killed process {pid}" in output_lower and ("out of memory" in output_lower or "oom killer" in output_lower):
        return "被系统杀掉 (例如 OOM)"
    else:
        return "正常退出"

# ======================
# 🧠 主监控逻辑
# ======================
def monitor():
    print("🚀 正在监控 python/python3 进程...")
    tracked = {}

    while True:
        current = get_python_processes()
        current_pids = set(p.pid for p in current)

        # 跟踪新启动的进程
        for proc in current:
            if proc.pid not in tracked:
                try:
                    tracked[proc.pid] = {
                        'cmdline': ' '.join(proc.cmdline()),
                        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(proc.create_time()))
                    }
                    print(f"[+] 跟踪新进程 PID {proc.pid}")
                except Exception:
                    continue

        # 检查已经结束的进程
        ended_pids = [pid for pid in tracked if pid not in current_pids]
        for pid in ended_pids:
            info = tracked.pop(pid)
            reason = get_exit_reason(pid)
            subject = f"⚠️ Python进程 PID {pid} 已结束"
            body = f"""
进程信息:
----------
PID       : {pid}
命令行   : {info['cmdline']}
启动时间 : {info['start_time']}
结束时间 : {time.strftime('%Y-%m-%d %H:%M:%S')}
退出原因 : {reason}
            """.strip()
            send_email(subject, body)
        time.sleep(1)

if __name__ == '__main__':
    try:
        monitor()
    except KeyboardInterrupt:
        subject = "🛑 Python进程监控脚本被手动终止"
        body = f"""
监控脚本收到 KeyboardInterrupt (例如 Ctrl+C)。
终止时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
状态    : 手动中断脚本运行。
        """.strip()
        send_email(subject, body)
        print("👋 脚本手动终止，通知邮件已发送。")