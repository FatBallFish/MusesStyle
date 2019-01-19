from django.test import TestCase, Client
import json
import base64
# Create your tests here.


class Test(TestCase):
    # 测试函数执行前执行
    def setUp(self):
        print("======in setUp")

    def test_create_filter(self):
        with open("res/image/mosaic.jpg", "rb") as f:
            base64_data = base64.urlsafe_b64encode(f.read())
            base64_data = base64_data.decode()

        filter_info = json.dumps({
            'filter_name': '123',
            'owner': 'xnf',
            'state': 1,
            'style_template': base64_data,
            'brush_size': 512,
            'brush_intensity': 500,
            'smooth': 0
        })
        response = self.client.post('/api/createFilter', filter_info, content_type="application/json")
        self.assertEqual(response.status_code, 200) # 检查返回数据

    # 测试函数执行后执行
    def tearDown(self):
        print("======in tearDown")
