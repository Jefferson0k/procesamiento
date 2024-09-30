from django.urls import path
from .views import recognize_colores, recognize_numeros, recognize_prendas, recognize_saludos

urlpatterns = [
    # Rutas para el reconocimiento de diferentes categorías de acciones
    path('recognize-colores/', recognize_colores, name='recognize-colores'),
    path('recognize-numeros/', recognize_numeros, name='recognize-numeros'),
    path('recognize-prendas/', recognize_prendas, name='recognize-prendas'),
    path('recognize-saludos/', recognize_saludos, name='recognize-saludos'),
]
