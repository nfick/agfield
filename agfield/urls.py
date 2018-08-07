from django.conf.urls import include, url
from django.urls import path

from django.contrib import admin
admin.autodiscover()

import field.views

# Examples:
# url(r'^$', 'gettingstarted.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    url(r'^$', field.views.index, name='index'),
    url(r'^about', field.views.about, name='about'),
    url(r'^field/', include('field.urls')),
    path('admin/', admin.site.urls),
]
