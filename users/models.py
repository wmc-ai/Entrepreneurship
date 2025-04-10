from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    user_type = models.CharField(max_length=20, choices=[
        ('SEEKER', '求职者'),
        ('EMPLOYER', '雇主')
    ])
    phone = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)


class JobSeeker(models.Model):
    user = models.OneToOneField(UserProfile, on_delete=models.CASCADE)
    skills = models.TextField()  # 技能
    experience = models.TextField()  # 工作经验
    education = models.TextField()  # 教育背景


class Employer(models.Model):
    user = models.OneToOneField(UserProfile, on_delete=models.CASCADE)
    company_name = models.CharField(max_length=100)  # 公司名称
    company_description = models.TextField()  # 公司描述


from django.db import models

# Create your models here.
class Job(models.Model):
    employer = models.ForeignKey(Employer, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)  # 职位标题
    description = models.TextField()  # 职位描述
    salary = models.DecimalField(max_digits=10, decimal_places=2)  # 薪资
    working_hours = models.JSONField()  # 工作时间段
    location = models.CharField(max_length=100)  # 工作地点
    skills_required = models.JSONField()  # 所需技能
    created_at = models.DateTimeField(auto_now_add=True)  # 创建时间
    status = models.CharField(max_length=20, default='OPEN')

    def __str__(self):
        return self.title

class JobApplication(models.Model):
    job = models.ForeignKey(Job, on_delete=models.CASCADE)
    applicant = models.ForeignKey(JobSeeker, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=[
        ('APPLIED', 'Applied'),
        ('ACCEPTED', 'Accepted'),
        ('REJECTED', 'Rejected'),
        ('COMPLETED', 'Completed')
    ])
    user_feedback = models.IntegerField(null=True)  # 用户对推荐的反馈
    created_at = models.DateTimeField(auto_now_add=True)

class UserPreference(models.Model):
    user = models.OneToOneField(JobSeeker, on_delete=models.CASCADE)
    preferred_salary_range = models.JSONField(default=dict)  # 添加默认值
    preferred_locations = models.JSONField(default=list)  # 添加默认值
    preferred_working_hours = models.JSONField(default=list)  # 添加默认值
    skill_weights = models.JSONField(default=dict)  # 添加默认值
    interaction_history = models.JSONField(default=dict)  # 添加默认值