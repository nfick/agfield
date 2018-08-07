$(document).ready(function(){   

  function changeMarker(m){
    if (typeof marker !== "undefined" ){ 
      marker.remove();         
    }

    //add marker
    marker = new  mapboxgl.Marker({
        draggable: true
    })
        .setLngLat([m.lngLat.lng, m.lngLat.lat])
        .addTo(map);

    updateImage(marker);
  }

  function displayCoords(m) {
    var deffered = $.Deferred();
    lngLat = m.getLngLat();
    map.setBearing(0);
    map.setPitch(0);
    lngLatBounds = map.getBounds();
    swLngLat = lngLatBounds.getSouthWest();
    neLngLat = lngLatBounds.getNorthEast();
    coordinates.style.display = 'inline-block';
    coordinates.innerHTML = 'Longitude: ' + lngLat.lng + '<br />Latitude: ' + lngLat.lat;
      //+ '<br /> SW Longitude: ' + swLngLat.lng + '<br />SW Latitude: ' + swLngLat.lat 
      //+ '<br /> NE Longitude: ' + neLngLat.lng + '<br />NE Latitude: ' + neLngLat.lat;
    deffered.resolve(lngLat);
    return deffered.promise();
  } 

  function updateImage(m) {
    var promise = displayCoords(m);
    promise.then(function() { 
      img = map.getCanvas().toDataURL('image/png');
      //document.getElementById("image").innerHTML = img;
    });
  }

  mapboxgl.accessToken = mapbox_key;

  var map = new mapboxgl.Map({
    container: 'map',
    style: 'mapbox://styles/mapbox/satellite-v9',
    center: [-71.0691572, 42.3604266],
    zoom: 12,
    preserveDrawingBuffer: true
  });

  var marker;
  var lngLat;
  var img;

  map.addControl(new MapboxGeocoder({
    accessToken: mapboxgl.accessToken
  }));

  map.on('click', changeMarker);

  marker.on('dragend', updateImage);
});