from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^about', views.about, name='about'),
    url(r'^samples', views.samples, name='samples'),
    url(r'^findedges', views.findedges, name='findedges'),
    url(r'^twodlidar', views.twodlidar, name='twodlidar'),
    url(r'^step(?P<step_num>\d)/$', views.step, name='step'),
]
