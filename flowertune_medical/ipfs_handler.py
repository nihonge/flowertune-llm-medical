# flowertune_medical/ipfs_handler.py

import requests
import os
import tarfile
import json

class IPFSHandler:
    def __init__(self, host='127.0.0.1', port=5001):
        # IPFS 默认 API 地址
        self.api_url = f"http://{host}:{port}/api/v0"

    def upload_folder(self, folder_path: str) -> str | None:
        """
        上传文件夹到 IPFS (递归)。
        相当于命令行: ipfs add -r <folder>
        """
        try:
            # 1. 准备上传的文件列表
            # 我们需要构建一个符合 multipart/form-data 的列表
            files = []
            
            # 遍历文件夹
            if not os.path.exists(folder_path):
                print(f"❌ [IPFS] Folder not found: {folder_path}")
                return None

            base_folder_name = os.path.basename(os.path.normpath(folder_path))

            for root, dirs, filenames in os.walk(folder_path):
                for filename in filenames:
                    abs_path = os.path.join(root, filename)
                    # 计算相对路径，例如: "tmp_client_0/adapter_model.bin"
                    # IPFS 需要知道目录结构
                    rel_path = os.path.relpath(abs_path, os.path.dirname(folder_path))
                    
                    # 必须以 binary 模式打开
                    files.append(
                        ('file', (rel_path, open(abs_path, 'rb'), 'application/octet-stream'))
                    )

            if not files:
                print("❌ [IPFS] Folder is empty.")
                return None

            # 2. 调用 IPFS /add API
            # wrap-with-directory=false 因为我们在 rel_path 里已经包含了文件夹结构
            # pin=true 默认就会 pin 住
            params = {
                'recursive': 'true', 
                'wrap-with-directory': 'false', # 我们自己处理文件路径结构
                'quiet': 'false'
            }
            
            # 发送请求
            # 注意：文件多的时候这里可能会慢，生产环境可以用 stream 上传，这里简化处理
            response = requests.post(f"{self.api_url}/add", params=params, files=files)
            
            # 关闭文件句柄
            for _, (_, f, _) in files:
                f.close()

            response.raise_for_status()

            # 3. 解析响应
            # IPFS add 返回的是多行 JSON，每一行对应一个文件/文件夹
            # 类似:
            # {"Name":"folder/file.txt", "Hash":"..."}
            # {"Name":"folder", "Hash":"..."}  <-- 最后一个通常是根目录
            
            lines = response.text.strip().split('\n')
            
            # 我们需要找到代表整个文件夹的那一行
            # 通常是最后一行，且 Name 等于我们将要上传的文件夹名
            # 为了保险，我们找那个 Name 等于 base_folder_name 的
            
            root_cid = None
            # 倒序查找
            for line in reversed(lines):
                if not line: continue
                try:
                    obj = json.loads(line)
                    # 如果上传路径包含文件夹结构，API返回的Name通常就是文件夹名
                    if obj['Name'] == base_folder_name:
                        root_cid = obj['Hash']
                        break
                except:
                    pass
            
            # 如果没找到名字匹配的，就取最后一行（通常也是对的）
            if not root_cid and lines:
                root_cid = json.loads(lines[-1])['Hash']

            return root_cid

        except Exception as e:
            print(f"❌ [IPFS] Upload Failed: {e}")
            return None

    def download(self, cid: str, target_dir: str) -> bool:
        """
        从 IPFS 下载并解压。
        相当于命令行: ipfs get <cid>
        """
        try:
            # IPFS /get API 返回的是一个 TAR 归档流
            url = f"{self.api_url}/get"
            params = {'arg': cid}
            
            # stream=True 避免一次性把大文件读进内存
            with requests.post(url, params=params, stream=True) as r:
                r.raise_for_status()
                
                # 确保目标目录存在
                os.makedirs(target_dir, exist_ok=True)
                
                # 直接解压流
                # mode="r|*" 表示读取流形式的 tar
                try:
                    with tarfile.open(fileobj=r.raw, mode="r|*") as tar:
                        tar.extractall(path=target_dir)
                    return True
                except tarfile.ReadError:
                    # 有时候 IPFS 可能还没同步完，返回了错误数据
                    print(f"❌ [IPFS] Download stream is not a valid tar archive.")
                    return False
                    
        except Exception as e:
            print(f"❌ [IPFS] Download Failed: {e}")
            return False