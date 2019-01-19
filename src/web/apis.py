import base64

from src.web.models import Filter
from src.web.models import FilterState
from src.web.models import FilterResult
from src.web.serializers import FilterSerializer
from src.web.process import filter_queue
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
import json


@api_view(['POST'])
def create_filter(request):
    filter_data = json.loads(request.body) # 载入json数据
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
    result_id = int(request.GET.get("result_id"))
    filter_result = FilterResult.objects.filter(id=result_id)[0]
    image = filter_result.filter.style_template
    image = base64.urlsafe_b64decode(image)
    image = "data:image/png;base64,"+base64.b64encode(image).decode()
    data = json.dumps({'styleImage': image})
    return Response(data, status=status.HTTP_200_OK, content_type="application/json")