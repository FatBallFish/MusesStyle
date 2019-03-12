import sys
from multiprocessing import Process, Queue
from src.base.train import train
from datetime import datetime
from src.web.models import Filter, FilterState
from django.db import connections

filter_queue = Queue(maxsize=128)


def close_old_connections():
    for conn in connections.all():
        conn.close_if_unusable_or_obsolete()


def process_manage(queue: Queue, index):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    while True:
        print("等待用户数据")
        filter_info: Filter = queue.get()
        print("有新增的训练任务")

        close_old_connections()
        filter = Filter.objects.filter(id=filter_info.id)[0]
        filter.start_time = datetime.now()
        filter.state = FilterState.objects.filter(id=3)[0]
        filter.save()

        process = Process(target=train, args=(filter_info, True, ))
        process.start()
        process.join()

        close_old_connections()
        filter = Filter.objects.filter(id=filter_info.id)[0]
        filter.finish_time = datetime.now()
        filter.state = FilterState.objects.filter(id=4)[0]
        filter.save()
        print("滤镜训练完成")


def init():
    process_pool = []
    for index in range(1, 3):
        process = Process(target=process_manage, args=(filter_queue, index,))
        print("进程%d建立成功, 使用第%d块显卡"%(index, index))
        process_pool.append(process)
        process.start()
    filter_list = Filter.objects.filter(finish_time__isnull=True)
    for filter in filter_list:
        filter_queue.put(filter)
        print(filter)


if sys.argv[1] == 'runserver':
    init()

