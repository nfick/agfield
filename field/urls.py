from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^about', views.about, name='about'),
    url(r'^samples', views.samples, name='samples'),
    url(r'^findedges', views.findedges, name='findedges'),
    url(r'^twodlidar', views.twodlidar, name='twodlidar'),
    url(r'^step1', views.step1, name='step1'),
    url(r'^step2', views.step2, name='step2'),
    url(r'^step3', views.step3, name='step3'),
    url(r'^step4', views.step4, name='step4'),
    url(r'^step5', views.step5, name='step5'),
    url(r'^step6', views.step6, name='step6'),
]
