#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rpp_control_cli.py  (Auto-start version)

目标：在一个 Python 程序里完成：
1) 自动启动底盘驱动：
   ros2 launch rpp_ros_driver driver.launch.py port:=/dev/ttyUSB0 baud_rate:=230400
2) 进入交互控制台，你输入指令就能前进/后退/转向；执行完自动停；
3) 可以连续输入下一条指令，直到输入 exit/quit/q 退出。

说明：

- rpp_ros_driver 有 0.5s “无新 cmd_vel 自动刹车”机制，所以本脚本会持续 20Hz 发布 /cmd_vel。
- f/b/l/r 采用“开环计时”（按速度+时间换算距离/角度）。地面打滑会有误差；后续可升级成 /odom 闭环。
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional



# ---------------- ROS env helper ----------------

def _need_reexec_with_ros_env() -> bool:
    ap = os.environ.get("AMENT_PREFIX_PATH", "")
    return ("/opt/ros/humble" not in ap)

def _reexec_with_ros_env(argv: list[str]) -> None:
    """自动 source ROS2 环境后重新执行自己（让你在 PyCharm/SSH 里直接运行也能生效）"""
    this = os.path.abspath(argv[0])
    args = " ".join(shlex.quote(a) for a in argv[1:])
    cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source ~/rpp_ws/install/setup.bash && "
        "export ROS_DOMAIN_ID=0 && "
        "unset ROS_LOCALHOST_ONLY && "
        f"python3 {shlex.quote(this)} {args}"
    )
    os.execvp("bash", ["bash", "-lc", cmd])


try:
    if _need_reexec_with_ros_env():
        _reexec_with_ros_env(sys.argv)

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
except Exception as e:
    print("[ERROR] 导入 ROS2 Python 依赖失败。请在当前终端先执行：")
    print("  source /opt/ros/humble/setup.bash")
    print("  source ~/rpp_ws/install/setup.bash")
    print("然后再运行脚本。原始错误：", repr(e))
    sys.exit(1)


# ---------------- ROS publisher ----------------

@dataclass
class Vel:
    vx: float = 0.0  # m/s
    vy: float = 0.0  # m/s (全向底盘才有效；2wd 一般无效)
    wz: float = 0.0  # rad/s

class CmdVelPublisher:
    def __init__(self, topic: str = "/cmd_vel", hz: float = 20.0):
        self.topic = topic
        self.hz = hz
        self._lock = threading.Lock()
        self._vel = Vel(0.0, 0.0, 0.0)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        rclpy.init(args=None)
        self.node = Node("rpp_cmdvel_cli")
        self.pub = self.node.create_publisher(Twist, self.topic, 10)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_vel(self, vx: float, vy: float, wz: float):
        with self._lock:
            self._vel = Vel(vx, vy, wz)

    def stop_robot(self):
        self.set_vel(0.0, 0.0, 0.0)

    def shutdown(self):
        # 多发几次 0，确保底盘收到
        for _ in range(8):
            self._publish_once(0.0, 0.0, 0.0)
            time.sleep(0.02)
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.node.destroy_node()
        rclpy.shutdown()

    def _publish_once(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.pub.publish(msg)

    def _loop(self):
        period = 1.0 / float(self.hz)
        while self._running and rclpy.ok():
            with self._lock:
                v = self._vel
            self._publish_once(v.vx, v.vy, v.wz)
            time.sleep(period)


# ---------------- driver manager ----------------

class DriverManager:
    """
    只负责：启动/停止 rpp_ros_driver。
    - 会先检查系统里是否已经有 /rpp_ros_driver 在运行；有则不重复启动（避免“跑一次行，再跑不行”）。
    - 退出时默认只停止“本脚本启动的那份”。
    """
    def __init__(self, robot: str, port: str, baud: int, publish_odom: bool = True, publish_odom_tf: bool = True):
        self.robot = robot
        self.port = port
        self.baud = baud
        self.publish_odom = publish_odom
        self.publish_odom_tf = publish_odom_tf

        self.proc: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self.started_by_me: bool = False

    @staticmethod
    def _ros_node_exists(node_name: str) -> bool:
        # 通过 bash -lc 调 ros2 node list（保证 source 环境）
        cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source ~/rpp_ws/install/setup.bash && "
            "ros2 node list"
        )
        try:
            out = subprocess.check_output(["bash", "-lc", cmd], text=True, stderr=subprocess.STDOUT, timeout=3)
            return any(line.strip() == node_name for line in out.splitlines())
        except Exception:
            return False

    def start(self):
        # 系统已经有 /rpp_ros_driver -> 不重复启动
        if self._ros_node_exists("/rpp_ros_driver"):
            print("[INFO] 检测到 /rpp_ros_driver 已在运行，跳过启动（避免抢串口）")
            self.started_by_me = False
            return

        if self.proc and self.proc.poll() is None:
            print("[INFO] driver 已在运行（本脚本启动的）")
            self.started_by_me = True
            return

        cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source ~/rpp_ws/install/setup.bash && "
            "export ROS_DOMAIN_ID=0 && unset ROS_LOCALHOST_ONLY && "
            f"ros2 launch rpp_ros_driver driver.launch.py "
            f"robot:={shlex.quote(self.robot)} "
            f"port:={shlex.quote(self.port)} "
            f"baud_rate:={int(self.baud)} "
            f"publish_odom:={'true' if self.publish_odom else 'false'} "
            f"publish_odom_tf:={'true' if self.publish_odom_tf else 'false'}"
        )
        print("[INFO] 启动 driver：ros2 launch rpp_ros_driver ...")
        self.proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.started_by_me = True
        self._log_thread = threading.Thread(target=self._pipe_logs, daemon=True)
        self._log_thread.start()

    def stop(self, timeout: float = 5.0, force: bool = False):
        """
        默认只停止“本脚本启动的那份”；
        force=True 时，会额外尝试 pkill（谨慎用）。
        """
        if not force and not self.started_by_me:
            print("[INFO] driver 不是本脚本启动的，默认不停止（如需强制停止：stop_driver_force）")
            return

        if self.proc and self.proc.poll() is None:
            print("[INFO] 停止 driver（SIGINT）...")
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print("[WARN] driver 未在超时内退出，强制 SIGKILL")
                self.proc.kill()
            self.proc = None
            return

        if force:
            cmd = (
                "source /opt/ros/humble/setup.bash && "
                "source ~/rpp_ws/install/setup.bash && "
                "sudo pkill -f rpp_ros_driver_node || true"
            )
            subprocess.call(["bash", "-lc", cmd])

    def _pipe_logs(self):
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            print("[driver]", line.rstrip())


# ---------------- motion macros (open-loop) ----------------

def move_distance(pub: CmdVelPublisher, meters: float, speed: float):
    if speed <= 0:
        raise ValueError("speed 必须 > 0")
    duration = abs(meters) / speed
    vx = speed if meters >= 0 else -speed
    pub.set_vel(vx, 0.0, 0.0)
    time.sleep(duration)
    pub.stop_robot()
    time.sleep(0.2)

def rotate_deg(pub: CmdVelPublisher, deg: float, wz: float):
    if wz <= 0:
        raise ValueError("wz 必须 > 0")
    rad = abs(deg) * 3.141592653589793 / 180.0
    duration = rad / wz
    wz_cmd = wz if deg >= 0 else -wz
    pub.set_vel(0.0, 0.0, wz_cmd)
    time.sleep(duration)
    pub.stop_robot()
    time.sleep(0.2)


HELP = r"""
命令（在 rpp-cli> 后输入）：

  f meters [speed]         前进 meters 米（默认 speed=0.15 m/s）
  b meters [speed]         后退 meters 米（默认 speed=0.15 m/s）
  l deg [wz]               左转 deg 度（默认 wz=0.40 rad/s）
  r deg [wz]               右转 deg 度（默认 wz=0.40 rad/s）

  seq "f 0.5; l 90; f 0.2"  顺序执行（分号分隔），每条执行完自动停

  v vx vy wz               持续速度模式（一直走，直到你输入 zero）
  zero                     立即刹车（发布 0 速度）

  stop_driver              停止 driver（仅停止本脚本启动的那份）
  stop_driver_force        强制停止 driver（会尝试 pkill，谨慎用）
  status                   显示 driver 状态（本脚本启动的那份）

  exit / quit / q          停车并退出
  help                     显示帮助

注意：
- 2wd 底盘一般不支持 vy（横移），vy 会被忽略；“左右”通常指左转右转。
"""


def run_cli(args):
    driver = DriverManager(args.robot, args.port, args.baud, args.publish_odom, args.publish_odom_tf)
    pub = CmdVelPublisher(topic="/cmd_vel", hz=args.hz)
    pub.start()

    # 自动启动 driver（符合你“运行程序就自动启动 ros2 launch ...”的需求）
    if args.auto_start_driver:
        driver.start()
        time.sleep(1.0)

    print(HELP)

    try:
        while True:
            line = input("rpp-cli> ").strip()
            if not line:
                continue

            low = line.lower()
            if low in ("help", "?"):
                print(HELP); continue
            if low in ("exit", "quit", "q"):
                pub.stop_robot()
                if args.stop_driver_on_exit:
                    driver.stop(force=args.force_stop_driver_on_exit)
                break

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "status":
                running = (driver.proc is not None and driver.proc.poll() is None)
                print("[INFO] driver running (started_by_me) =", running, "(started_by_me=", driver.started_by_me, ")")
                continue
            if cmd == "stop_driver":
                driver.stop(force=False); continue
            if cmd == "stop_driver_force":
                driver.stop(force=True); continue
            if cmd == "zero":
                pub.stop_robot(); print("[OK] stop"); continue
            if cmd == "v":
                if len(parts) != 4:
                    print("用法：v vx vy wz"); continue
                vx, vy, wz = map(float, parts[1:])
                pub.set_vel(vx, vy, wz)
                print(f"[OK] set vel vx={vx} vy={vy} wz={wz}")
                continue

            if cmd in ("f", "b"):
                if len(parts) < 2:
                    print("用法：f meters [speed]  或  b meters [speed]")
                    continue
                meters = float(parts[1])
                meters = abs(meters) if cmd == "f" else -abs(meters)
                speed = float(parts[2]) if len(parts) >= 3 else 0.15
                print(f"[DO] move {meters:.3f} m @ {speed:.3f} m/s")
                move_distance(pub, meters, speed)
                print("[OK] done")
                continue

            if cmd in ("l", "r"):
                if len(parts) < 2:
                    print("用法：l deg [wz]  或  r deg [wz]")
                    continue
                deg = float(parts[1])
                deg = abs(deg) if cmd == "l" else -abs(deg)
                wz = float(parts[2]) if len(parts) >= 3 else 0.40
                print(f"[DO] rotate {deg:.1f} deg @ {wz:.3f} rad/s")
                rotate_deg(pub, deg, wz)
                print("[OK] done")
                continue

            if cmd == "seq":
                raw = line[len("seq"):].strip()
                if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                    raw = raw[1:-1]
                steps = [s.strip() for s in raw.split(";") if s.strip()]
                for s in steps:
                    print("[SEQ]", s)
                    p = s.split()
                    if not p:
                        continue
                    c = p[0].lower()
                    if c == "zero":
                        pub.stop_robot(); time.sleep(0.2)
                    elif c in ("f", "b"):
                        meters = float(p[1])
                        meters = abs(meters) if c == "f" else -abs(meters)
                        speed = float(p[2]) if len(p) >= 3 else 0.15
                        move_distance(pub, meters, speed)
                    elif c in ("l", "r"):
                        deg = float(p[1])
                        deg = abs(deg) if c == "l" else -abs(deg)
                        wz = float(p[2]) if len(p) >= 3 else 0.40
                        rotate_deg(pub, deg, wz)
                    elif c == "v":
                        vx, vy, wz = map(float, p[1:])
                        pub.set_vel(vx, vy, wz)
                    else:
                        print("[WARN] seq 不支持命令：", c)
                print("[OK] seq done")
                continue

            print("[WARN] 未识别命令，输入 help 查看用法。")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C")
    finally:
        pub.stop_robot()
        if args.stop_driver_on_exit:
                    driver.stop(force=args.force_stop_driver_on_exit)
        pub.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="cr100")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=230400)
    ap.add_argument("--hz", type=float, default=20.0, help="发布 /cmd_vel 的频率")
    ap.add_argument("--auto-start-driver", action="store_true", default=True,
                    help="运行程序时自动启动 driver（默认开启）")
    ap.add_argument("--no-auto-start-driver", action="store_false", dest="auto_start_driver",
                    help="关闭自动启动 driver")
    ap.add_argument("--stop-driver-on-exit", action="store_true", default=True,
                    help="退出时停止 driver（仅停止本脚本启动的那份）")
    ap.add_argument("--force-stop-driver-on-exit", action="store_true", default=False,
                    help="退出时强制停止 driver（会尝试 pkill，占串口时可用，谨慎）")
    ap.add_argument("--publish-odom", action="store_true", default=True)
    ap.add_argument("--publish-odom-tf", action="store_true", default=True)
    args = ap.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
