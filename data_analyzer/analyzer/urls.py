from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('download/<str:analysis_type>/<str:file_name>/', views.download_analysis_report, name='download_analysis_report'),
]