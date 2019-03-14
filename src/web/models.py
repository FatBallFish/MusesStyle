from io import BytesIO
from django.db import models
from django.contrib.auth.models import User
from django.utils.html import format_html
from src.web.views import html_modal, html_item, html_js
from PIL import Image
import base64

# Create your models here.


class FilterState(models.Model):
    """ state:
        1 -- added
        2 -- queuing
        3 -- training
        4 -- ready
        5 -- published
    """
    state = models.CharField(max_length=10, verbose_name="滤镜状态")

    def __str__(self):
        return self.state

    class Meta:
        verbose_name = '滤镜状态'
        verbose_name_plural = verbose_name


class FilterResult(models.Model):
    image1 = models.TextField(null=True, verbose_name="风格结果1")
    image2 = models.TextField(null=True, verbose_name="风格结果2")
    image3 = models.TextField(null=True, verbose_name="风格结果3")
    image4 = models.TextField(null=True, verbose_name="风格结果4")
    image5 = models.TextField(null=True, verbose_name="风格结果5")
    image6 = models.TextField(null=True, verbose_name="风格结果6")
    image7 = models.TextField(null=True, verbose_name="风格结果7")
    image8 = models.TextField(null=True, verbose_name="风格结果8")
    image9 = models.TextField(null=True, verbose_name="风格结果9")
    image10 = models.TextField(null=True, verbose_name="风格结果10")

    def image_data(self):

        def center_crop(image, x, y):
            width, height = image.size[0], image.size[1]
            crop_side = min(width, height)
            width_crop = (width-crop_side)//2
            height_crop = (height-crop_side)//2
            box = (width_crop, height_crop, width_crop+crop_side, height_crop+crop_side)
            image = image.crop(box)
            image = image.resize((x, y), Image.ANTIALIAS)
            return image

        style_template = Image.open(BytesIO(base64.urlsafe_b64decode(self.filter.style_template)))
        style_template = center_crop(style_template, 256, 256)
        output_buffer = BytesIO()
        style_template.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        style_template = base64.urlsafe_b64encode(byte_data).decode()
        image_list = [style_template, self.image1, self.image2, self.image3, self.image4, self.image5,
                      self.image6, self.image7, self.image8, self.image9, self.image10]
        result_list = []

        for image_result in image_list:
            try:
                image = Image.open(BytesIO(base64.urlsafe_b64decode(image_result)))
                width, height = image.size[0], image.size[1]
                scale = 256 / height
                image = image.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
                output_buffer = BytesIO()
                image.save(output_buffer, format='JPEG')
                byte_data = output_buffer.getvalue()
                result_list.append("data:image/png;base64,"+base64.b64encode(byte_data).decode())
            except:
                result_list.append("图像处理出错")

        html_code = html_modal % (self.id, "风格模板："+self.filter.filter_name+","+str(self.filter.upload_id), self.id)
        for idx, result in enumerate(result_list):
            html_code += html_item % (self.id, idx, self.id, result, self.id)
        html_code += html_js.replace("{", "{{").replace("}", "}}") % (self.id, self.id)
        return format_html(html_code)
    image_data.short_description = u'滤镜训练结果'

    class Meta:
        verbose_name = '滤镜训练结果'
        verbose_name_plural = verbose_name


class Filter(models.Model):
    """ basic information of art filters
    owner:
    if owner != null:
        the filter is a shared filter
    """
    upload_id = models.IntegerField(null=True, unique=True, auto_created=True, verbose_name="唯一标识")
    filter_name = models.CharField(max_length=30, verbose_name="滤镜名称")
    owner = models.CharField(null=True, max_length=30, verbose_name="创建者")
    user_id = models.IntegerField(null=True, verbose_name="用户id")
    state = models.ForeignKey(to=FilterState, on_delete=models.CASCADE, verbose_name="状态")
    style_template = models.TextField(null=True, verbose_name="风格模板")
    brush_size = models.IntegerField(default=512, verbose_name="笔刷大小")
    brush_intensity = models.IntegerField(default=512, verbose_name="风格强度")
    smooth = models.IntegerField(default=0, verbose_name="平滑度")
    upload_time = models.DateTimeField(null=True, blank=True, verbose_name="上传时间")
    upload_day = models.CharField(max_length=20, null=True, blank=True, verbose_name="上传日期")
    start_time = models.DateTimeField(null=True, blank=True, verbose_name="开始训练时间")
    finish_time = models.DateTimeField(null=True, blank=True, verbose_name="结束训练时间")
    schedule = models.DecimalField(null=True, blank=True, max_digits=6, decimal_places=2,
                                   default=0, verbose_name="训练进度")
    result = models.OneToOneField(to=FilterResult, null=True, on_delete=models.CASCADE, verbose_name="训练结果")
    thumbnail = models.TextField(null=True, verbose_name="风格模板缩略图")

    def image_data(self):
        style_template = self.thumbnail
        if style_template.startswith("data:image/png;base64,"):
            style_template = style_template[22:]
        image = Image.open(BytesIO(base64.urlsafe_b64decode(style_template)))
        image = image.convert('RGB')
        width, height = image.size[0], image.size[1]
        scale = 80 / height
        image = image.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        return format_html(
            '<img src="{}" height="80px"/>',
            "data:image/png;base64,"+base64.b64encode(byte_data).decode(),
        )
    image_data.short_description = u'风格模板'

    def __str__(self):
        return self.filter_name

    class Meta:
        verbose_name = '滤镜信息'
        verbose_name_plural = verbose_name



