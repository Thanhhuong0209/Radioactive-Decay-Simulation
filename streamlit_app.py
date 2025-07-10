import sys
import os

# Đảm bảo đường dẫn src là tuyệt đối
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import dashboard
 