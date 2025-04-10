from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .models import UserProfile, JobSeeker, Employer, UserPreference, Job, JobApplication
from .matching import job_matcher
from datetime import datetime


def home(request):
    return render(request, 'users/home.html')


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user_type = request.POST.get('user_type')
        phone = request.POST.get('phone')

        # 创建用户
        user = User.objects.create_user(username=username, password=password)

        # 创建用户档案
        profile = UserProfile.objects.create(
            user=user,
            user_type=user_type,
            phone=phone
        )

        # 根据用户类型创建具体角色
        if user_type == 'SEEKER':
            JobSeeker.objects.create(user=profile)
        else:
            Employer.objects.create(
                user=profile,
                company_name=request.POST.get('company_name')
            )

        return redirect('login')

    return render(request, 'users/register.html')


def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')

    return render(request, 'users/login.html')


@login_required
def set_preferences(request):
    if request.user.userprofile.user_type != 'SEEKER':
        return redirect('home')

    if request.method == 'POST':
        job_seeker = request.user.userprofile.jobseeker

        # 获取并处理表单数据
        min_salary = request.POST.get('min_salary', '0')
        max_salary = request.POST.get('max_salary', '0')

        # 确保薪资值有效
        try:
            min_salary = float(min_salary)
            max_salary = float(max_salary)
        except ValueError:
            min_salary = 0
            max_salary = 100000

        # 创建JSON格式的薪资范围
        salary_range = {
            'min': min_salary,
            'max': max_salary
        }

        # 处理位置和时间
        locations_text = request.POST.get('locations', '')
        locations = [loc.strip() for loc in locations_text.split(',') if loc.strip()]

        working_hours = request.POST.getlist('working_hours')
        if not working_hours:  # 如果没有选择，提供默认值
            working_hours = ['morning', 'afternoon', 'evening']

        # 处理技能
        skills_text = request.POST.get('skills', '')
        skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]

        # 确保技能不为空
        if not skills:
            skills = ['general']

        # 创建或更新用户偏好
        preference, created = UserPreference.objects.get_or_create(user=job_seeker)

        # 确保所有JSONField都有有效的JSON值
        preference.preferred_salary_range = salary_range
        preference.preferred_locations = locations
        preference.preferred_working_hours = working_hours
        preference.skill_weights = {skill: 1.0 for skill in skills}

        # 初始化交互历史（如果是新创建的）
        if created or not preference.interaction_history:
            preference.interaction_history = {
                'viewed_jobs': [],
                'applied_jobs': [],
                'completed_jobs': [],
                'feedback_history': []
            }

        preference.save()
        return redirect('job_recommendations')

    return render(request, 'users/preferences.html')
# 新增的职位推荐相关视图函数
@login_required
def job_recommendations(request):
    """职位推荐视图"""
    if request.user.userprofile.user_type != 'SEEKER':
        return redirect('home')

    # 获取求职者和可用职位
    job_seeker = request.user.userprofile.jobseeker
    available_jobs = Job.objects.filter(status='OPEN')

    # 获取推荐职位
    recommended_jobs = job_matcher.recommend_jobs(job_seeker, available_jobs)

    # 更新用户交互历史 - 记录查看推荐
    try:
        preference = UserPreference.objects.get(user=job_seeker)
        history = preference.interaction_history or {}

        if 'viewed_recommendations' not in history:
            history['viewed_recommendations'] = []

        history['viewed_recommendations'].append({
            'timestamp': str(datetime.now()),
            'job_ids': [job.id for job in recommended_jobs]
        })

        preference.interaction_history = history
        preference.save()
    except UserPreference.DoesNotExist:
        pass

    return render(request, 'users/recommendations.html', {
        'jobs': recommended_jobs
    })


@login_required
def job_detail(request, job_id):
    """职位详情视图"""
    job = Job.objects.get(id=job_id)

    # 记录用户查看行为
    if request.user.userprofile.user_type == 'SEEKER':
        job_seeker = request.user.userprofile.jobseeker

        try:
            preference = UserPreference.objects.get(user=job_seeker)
            history = preference.interaction_history or {}

            if 'viewed_jobs' not in history:
                history['viewed_jobs'] = []

            history['viewed_jobs'].append({
                'job_id': job_id,
                'timestamp': str(datetime.now())
            })

            preference.interaction_history = history
            preference.save()

            # 更新Q值 - 查看操作奖励为0.1
            job_matcher.update_q_value(job_seeker, job, 'view', 0.1)
        except UserPreference.DoesNotExist:
            pass

    return render(request, 'users/job_detail.html', {'job': job})


@login_required
def apply_job(request, job_id):
    """申请职位视图"""
    if request.user.userprofile.user_type != 'SEEKER':
        return redirect('home')

    job = Job.objects.get(id=job_id)
    job_seeker = request.user.userprofile.jobseeker

    # 创建申请记录
    JobApplication.objects.create(
        job=job,
        applicant=job_seeker,
        status='APPLIED'
    )

    # 更新用户交互历史
    try:
        preference = UserPreference.objects.get(user=job_seeker)
        history = preference.interaction_history or {}

        if 'applied_jobs' not in history:
            history['applied_jobs'] = []

        history['applied_jobs'].append({
            'job_id': job_id,
            'timestamp': str(datetime.now())
        })

        preference.interaction_history = history
        preference.save()

        # 更新Q值 - 申请操作奖励为0.5
        job_matcher.update_q_value(job_seeker, job, 'apply', 0.5)
    except UserPreference.DoesNotExist:
        pass

    return redirect('job_recommendations')


@login_required
def job_feedback(request, job_id):
    """职位反馈视图"""
    if request.method == 'POST':
        job = Job.objects.get(id=job_id)
        job_seeker = request.user.userprofile.jobseeker
        feedback_score = int(request.POST.get('feedback', 3))  # 默认为3 (中等)

        # 更新用户交互历史
        try:
            preference = UserPreference.objects.get(user=job_seeker)
            history = preference.interaction_history or {}

            if 'feedback_history' not in history:
                history['feedback_history'] = []

            history['feedback_history'].append({
                'job_id': job_id,
                'score': feedback_score,
                'timestamp': str(datetime.now())
            })

            # 计算平均反馈分数
            feedback_scores = [f['score'] for f in history['feedback_history']]
            history['average_feedback'] = sum(feedback_scores) / len(feedback_scores)

            preference.interaction_history = history
            preference.save()

            # 更新Q值 - 根据反馈分数计算奖励
            normalized_reward = (feedback_score - 1) / 4  # 映射到0-1范围
            job_matcher.update_q_value(job_seeker, job, 'feedback', normalized_reward)
        except UserPreference.DoesNotExist:
            pass

        return redirect('job_recommendations')

    return redirect('job_detail', job_id=job_id)


@login_required
def save_job(request, job_id):
    """保存职位视图"""
    if request.user.userprofile.user_type != 'SEEKER':
        return redirect('home')

    job = Job.objects.get(id=job_id)
    job_seeker = request.user.userprofile.jobseeker

    # 更新用户交互历史
    try:
        preference = UserPreference.objects.get(user=job_seeker)
        history = preference.interaction_history or {}

        if 'saved_jobs' not in history:
            history['saved_jobs'] = []

        # 检查是否已保存
        job_already_saved = any(s.get('job_id') == job_id for s in history['saved_jobs'])

        if not job_already_saved:
            history['saved_jobs'].append({
                'job_id': job_id,
                'timestamp': str(datetime.now())
            })

            preference.interaction_history = history
            preference.save()

            # 更新Q值 - 保存操作奖励为0.3
            job_matcher.update_q_value(job_seeker, job, 'save', 0.3)
    except UserPreference.DoesNotExist:
        pass

    return redirect('job_detail', job_id=job_id)