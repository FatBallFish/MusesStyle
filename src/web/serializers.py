from rest_framework import serializers
from src.web.models import Filter
from src.web.models import FilterState
from src.web.models import FilterResult
from datetime import datetime, date


class FilterSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    user_id = serializers.IntegerField() 
    filter_name = serializers.CharField(max_length=30)               # 滤镜名称
    owner = serializers.CharField(max_length=30, default='user', required=False)    # 滤镜所有者
    style_template = serializers.CharField()                         # 风格模板
    brush_size = serializers.IntegerField(default=512)               # 笔刷尺寸
    brush_intensity = serializers.IntegerField(default=512)          # 笔刷强度
    smooth = serializers.IntegerField(default=0)
    upload_time = serializers.DateTimeField(required=False)          # 上传时间
    upload_day = serializers.DateTimeField(required=False)           # 上传时间
    start_time = serializers.DateTimeField(required=False)           # 开始训练时间
    finish_time = serializers.DateTimeField(required=False)          # 完成训练时间

    def create(self, validated_data):
        return Filter.objects.create(
            state=FilterState.objects.filter(id=1)[0],
            result=None,
            upload_time=datetime.now(),
            upload_day=date.today().strftime('%Y-%m-%d'),
            **validated_data
        )

    def update(self, instance, validated_data):
        instance.filter_name = validated_data.get('filter_name', instance.filter_name)
        instance.state = validated_data.get('state', instance.state)
        instance.start_time = validated_data.get('start_time', instance.start_time)
        instance.finish_time = validated_data.get('finish_time', instance.finish_time)
        instance.save()
        return instance



