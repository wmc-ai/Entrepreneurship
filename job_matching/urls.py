from django.contrib import admin
from django.urls import path
from users import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('preferences/', views.set_preferences, name='preferences'),

    # 新增以下路由
    path('recommendations/', views.job_recommendations, name='job_recommendations'),
    path('job/<int:job_id>/', views.job_detail, name='job_detail'),
    path('job/<int:job_id>/apply/', views.apply_job, name='apply_job'),
    path('job/<int:job_id>/save/', views.save_job, name='save_job'),
    path('job/<int:job_id>/feedback/', views.job_feedback, name='job_feedback'),
]