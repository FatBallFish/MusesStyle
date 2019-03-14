import base64
from django.core import serializers
from src.web.models import Filter
from src.web.models import FilterState
from src.web.models import FilterResult
from src.web.serializers import FilterSerializer
from src.web.process import filter_queue
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.db.models import Q
import json
from PIL import Image
from io import BytesIO


def add_thumbnail(filter_data):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(filter_data["style_template"])))
    image = image.convert('RGB')
    width, height = image.size[0], image.size[1]
    scale = 512/height
    image = image.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
    output_buffer = BytesIO()
    image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    filter_data["thumbnail"] = base64.b64encode(byte_data).decode()
    return filter_data

@api_view(['POST'])
def create_filter(request):
    filter_data = json.loads(request.body) # 载入json数据
    filter_data = add_thumbnail(filter_data)
    serializer = FilterSerializer(data=filter_data) # 反序列化
    if serializer.is_valid():
        serializer.save() # 保存到数据库
        filter_info: Filter = serializer.instance
        filter_queue.put(filter_info)
        filter_info.state = FilterState.objects.filter(id=2)[0]
        print("添加滤镜至队列")
        # 加入进程队列
        return Response(status=status.HTTP_200_OK)
    else:
        print("数据非法")
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_image(request):
    result_id = int(request.GET.get("result_id"))
    image_id = int(request.GET.get("image_id"))
    filter_result = FilterResult.objects.filter(id=result_id)[0]
    if image_id == 0:
        image = filter_result.filter.style_template
    else:
        image = getattr(filter_result, "image"+str(image_id))
    image = base64.urlsafe_b64decode(image)
    image = "data:image/png;base64,"+base64.b64encode(image).decode()
    data = json.dumps({'styleImage': image})
    return Response(data, status=status.HTTP_200_OK, content_type="application/json")


@api_view(['GET'])
def get_image_list(request):
    result_id = int(request.GET.get("id"))
    filter_result = Filter.objects.filter(id=id)[0].result
    image = filter_result.filter.style_template
    image = base64.urlsafe_b64decode(image)
    image = "data:image/png;base64,"+base64.b64encode(image).decode()
    data = json.dumps({'styleImage': image})
    return Response(data, status=status.HTTP_200_OK, content_type="application/json")


@api_view(['GET'])
def get_filter_list(request):
    user_id = int(request.GET.get("user_id"))
    filters = Filter.objects.filter(Q(user_id=user_id) and Q(schedule__lt=100))
    print(filters)
    data = {
        "code": "OK",
        "message": "加载正在训练滤镜列表成功",
        "data": []
    }
    for filter in filters:
        # image = Image.open(BytesIO(base64.urlsafe_b64decode(filter.style_template)))
        # image = image.convert('RGB')
        # width, height = image.size[0], image.size[1]
        # scale = 512/height
        # image = image.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
        # output_buffer = BytesIO()
        # image.save(output_buffer, format='JPEG')
        # byte_data = output_buffer.getvalue()
        data["data"].append({
            "id": filter.id,
            "style_template": filter.thumbnail,
            "name": filter.filter_name,
            "state": filter.state.id,
            "schedule": float(filter.schedule)
        })
    json_data = json.dumps(data)
    return Response(json_data, status=status.HTTP_200_OK, content_type="application/json")

