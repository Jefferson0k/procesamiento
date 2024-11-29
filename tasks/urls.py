from django.urls import path
from .views import recognize_colores, recognize_decisiones, recognize_dias, recognize_numeros,recognize_saludos,recognize_pronombres

urlpatterns = [
    # Rutas para el reconocimiento de diferentes categorías de acciones
    path('recognize-colores/', recognize_colores, name='recognize-colores'),
    path('recognize-decisiones/', recognize_decisiones, name='recognize-decisiones'),
    path('recognize-dias/', recognize_dias, name='recognize-dias'),
    path('recognize-numeros/', recognize_numeros, name='recognize-numeros'),
    path('recognize-saludos/', recognize_saludos, name='recognize-saludos'),
    path('recognize-pronombres/', recognize_pronombres, name='recognize-pronombres'),
]
