from .views import *
from django.urls import path


urlpatterns = [
    path('', home, name='home'),
    path('register/', RegisterUser.as_view(), name='register'),
    path('login/', LoginUser.as_view(), name='login'),
    path('logout/', logout_user, name='logout'),
    path('addrecord/', AddRecord.as_view(), name='addrecord')
]