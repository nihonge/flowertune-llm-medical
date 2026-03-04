#!/bin/bash

echo "====================================================="
echo "  ⚠️  警告：准备执行 IPFS 核弹级清理协议！"
echo "  这将会物理抹除本地所有 IPFS 缓存文件，并重启节点。"
echo "====================================================="

# 给个反悔的机会
read -p "确定要引爆吗？(y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "🛑 指令已取消，硬盘保住了。"
    exit 1
fi

echo "💣 [1/4] 正在强行终止 IPFS 后台进程..."
pkill -x ipfs
sleep 2 # 等待进程彻底死亡

echo "🗑️  [2/4] 正在执行物理抹除 (rm -rf ~/.ipfs) ..."
rm -rf ~/.ipfs

echo "🌱 [3/4] 正在重新初始化纯净的 IPFS 节点..."
ipfs init > /dev/null 2>&1

echo "🚀 [4/4] 正在后台静默重启 IPFS Daemon..."
nohup ipfs daemon > ipfs_log.txt 2>&1 &
sleep 3 # 给守护进程一点启动时间

echo "====================================================="
echo "✅ 核弹清理完成！几 GB 的大模型垃圾已灰飞烟灭！"
echo "🔍 验证当前存活状态："
ipfs id | grep "ID"
echo "====================================================="
