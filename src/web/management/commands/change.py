from django.core.management.base import BaseCommand, CommandError

from src.web.models import Filter


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '-f',
            '--function',
            action='store',
            dest='function',
            default='close',
            help='name of author.',
        )

    def handle(self, *args, **options):
        try:
            exec("self."+options['function']+"()")
        except Exception as ex:
            self.stdout.write(self.style.ERROR('命令执行出错'))

    def intensity(self):
        self.stdout.write(self.style.SUCCESS("即将批量修改所有未训练滤镜的风格强度"))
        brush_intensity = int(input("请输入风格强度:"))
        flag = input("确认(yes or no):")
        if flag.startswith("y"):
            filter_list = Filter.objects.filter(finish_time__isnull=True)
            for filter_item in filter_list:
                self.stdout.write(self.style.SUCCESS("滤镜(%s)的风格强度从%d修改到%d"%(filter_item.filter_name,
                                                      filter_item.brush_intensity,brush_intensity)))
                filter_item.brush_intensity = brush_intensity
                filter_item.save()
        else:
            print("取消成功")
