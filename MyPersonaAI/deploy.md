* 运行server
```bash
# 安装依赖
pip install -r requirements.txt
# 安装ffmpeg
sudo apt install ffmpeg
# 启动
python main.py
```
* 运行web
```bash
cd web
# 使用高性能的npm
npm install -g pnpm
# 安装依赖
npm install
# 编译发布版本
npm run build
# 启动
npm run start