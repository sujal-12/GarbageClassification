from django.contrib import admin
from django.urls import path
from app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('',views.index,name='index'),
    path("registration", views.registration, name="registration"),
    path("loginform", views.loginform, name="loginform"),
    path("dashboard", views.dashboard, name="dashboard"),
    path('upload', views.upload_image, name='upload_image'),
    path('results', views.results, name='results'),
    path('download-pdf', views.download_pdf, name='download_pdf'),
    path('start_camera_view', views.start_camera_view, name='start_camera_view'),
    path('camera', views.start_camera_view, name='start_camera_view'),
    path('camera/feed', views.camera_feed, name='camera_feed'),
    path('camera/goto-results', views.goto_results_from_camera, name='goto_results_from_camera'),

] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)