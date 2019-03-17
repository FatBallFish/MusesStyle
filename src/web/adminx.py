import xadmin
from .models import Filter
from .models import FilterState
from .models import FilterResult
from xadmin import views


class FilterAdmin(object):
    list_display = ['id', 'filter_name', 'user_id', 'state', 'brush_size', 'brush_intensity', 'smooth',
                    'upload_time', 'start_time', 'finish_time', 'image_data', 'schedule']
    search_fields = ['filter_name', 'owner', 'state']
    list_editable = ['filter_name', 'owner', "user_id"]
    list_filter = ['filter_name', 'owner', 'state', 'brush_size', 'brush_intensity',
                   'smooth', 'upload_time', 'start_time', 'finish_time']
    data_charts = {
        "filter_count": {'title': u"滤镜数量统计",
                         "x-field": "upload_day",
                         "y-field": ("upload_day",),
                         'option': {
                             "series": {"bars": {"align": "center", "barWidth": 0.8, "show": True}},
                             "xaxis": {"aggregate": "count", "mode": "categories"}
                         }},
    }
    list_per_page = 20


class FilterStateAdmin(object):
    list_display = ['id', 'state']
    search_fields = ['state']
    list_editable = ['state']
    list_filter = ['state']


class FilterResultAdmin(object):
    list_display = ['id', 'image_data']
    search_fields = ['id']
    list_editable = ['id']
    list_filter = ['id']
    list_per_page = 5


class BaseSetting(object):
    enable_themes = True
    use_bootswatch = True


class GlobalSettings(object):
    site_title = "Muses自动训练系统"
    site_footer = "Muses自动训练系统"


xadmin.site.register(views.BaseAdminView, BaseSetting)
xadmin.site.register(views.CommAdminView, GlobalSettings)
xadmin.site.register(Filter, FilterAdmin)
xadmin.site.register(FilterState, FilterStateAdmin)
xadmin.site.register(FilterResult, FilterResultAdmin)
