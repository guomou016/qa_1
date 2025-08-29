from flask import Flask, jsonify
import os
import sys
from flask_cors import CORS

# 将项目根目录添加到Python路径，以便导入cs_data
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.insert(0, project_root)

from data.fetch_data import get_item_data

app = Flask(__name__)
CORS(app)  # 在Flask应用中启用CORS

@app.route('/item/<int:item_id>', methods=['GET'])
def get_item(item_id):
    """
    根据item_id获取相关数据
    ---1
    parameters:
      - name: item_id
        in: path
        type: integer
        required: true
        description: 要查询的项目ID
    responses:
      200:
        description: 成功获取数据
        schema:
          type: array
          items:
            type: object
            properties:
              doc_name:
                type: string
              img_path:
                type: string
              annex_name:
                type: string
              annex_path:
                type: string
              table:
                type: array
                items:
                  type: object
      404:
        description: 未找到对应的item_id
      500:
        description: 服务器内部错误
    """
    try:
        result = get_item_data(item_id)
        if not result:
            return jsonify({'message': f'未找到ID为 {item_id} 的项目数据'}), 404
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 在开发模式下运行，生产环境请使用Gunicorn或类似的WSGI服务器
    app.run(debug=True, host='0.0.0.0', port=5000)
