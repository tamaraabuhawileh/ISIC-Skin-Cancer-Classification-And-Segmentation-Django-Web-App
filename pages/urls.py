from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from . import views
urlpatterns = [
    path('' ,views.index ,name='index'),
    path('segmentation' ,views.segmentation ,name='segmentation'),
    path('classification' ,views.classification ,name='classification'),
    path('about' ,views.about ,name='about'),
]
