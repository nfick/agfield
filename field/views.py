from django.http import HttpResponse
from django.template import loader
from django.conf import settings
#import os

# Create your views here.
def index(request):
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        print(request.POST)
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

def step1(request):
    template = loader.get_template('field/step1.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def step2(request):
    template = loader.get_template('field/step2.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def step3(request):
    template = loader.get_template('field/step3.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def step4(request):
    template = loader.get_template('field/step4.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def step5(request):
    template = loader.get_template('field/step5.html')
    context = {
    }
    return HttpResponse(template.render(context, request))

def step6(request):
    template = loader.get_template('field/step6.html')
    context = {
    }
    return HttpResponse(template.render(context, request))