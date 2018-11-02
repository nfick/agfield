from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.conf import settings
from .models import FindMap
from .models import GraphWeather
import simplejson as json
#import os

# Create your views here.
def index(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request
        items = [(key, value) for key, value in request.POST.items()]
        image = request.POST.get('image')
        #print('\n'.join(image))
        coords = request.POST.get('coordinates_hidden')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        # print('Start date:', start_date, 'End date:', end_date, sep='\n')
        graphs = GraphWeather(start_date, end_date, coords)
        pcpn_graph = graphs.get_pcpn_graph()
        gdd_graph = graphs.get_gdd_graph()
        cdd_graph = graphs.get_cdd_graph()
        # print(graph)
        find = FindMap(coords, image)
        polygon = find.get_geojson_polygon()

        data = {'polygon': polygon, 'pcpn_graph': pcpn_graph, 
                'gdd_graph': gdd_graph,' cdd_graph': cdd_graph}
        #print(polygon)
        return JsonResponse(data)
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