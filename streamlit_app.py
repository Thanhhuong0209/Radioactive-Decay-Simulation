import sys
import os

# Thêm src vào path để import các module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import dashboard (toàn bộ giao diện sẽ chạy như cũ)
import dashboard 