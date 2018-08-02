from django.http import HttpResponse
from django.template import loader
from django.conf import settings
#import os

# Create your views here.
def index(request):
    template = loader.get_template('field/index.html')
    context = {
        'mapbox_key' : settings.MAPBOX_API_KEY
    }
    return HttpResponse(template.render(context, request))
