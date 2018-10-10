from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.conf import settings
from .models  import FindMap
#import os

# Create your views here.
def index(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request
        items = [(key, value) for key, value in request.POST.items()]
        image = request.POST.get('image')
        #print('\n'.join(image))
        coords = request.POST.get('coordinates_hidden')
        find = FindMap(coords, image)
        polygon = find.get_geojson_polygon()
        #print(polygon)
        return JsonResponse(polygon)
    else:
        template = loader.get_template('field/index.html')
        context = {
            'mapbox_key' : settings.MAPBOX_API_KEY
        }
        return HttpResponse(template.render(context, request))

def about(request):
    template = loader.get_template('field/about.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def samples(request):
    template = loader.get_template('field/samples.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def findedges(request):
    template = loader.get_template('field/findedges.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def twodlidar(request):
    template = loader.get_template('field/twodlidar.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def step(request, step_num):
    template = loader.get_template('field/step{}.html'.format(step_num))
    context = {
    }
    return HttpResponse(template.render(context, request))