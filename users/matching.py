import numpy as np
import spacy
import json
from sklearn.preprocessing import StandardScaler
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .models import Job, JobSeeker, UserPreference, JobApplication
from datetime import datetime

# 加载spaCy模型
try:
    nlp = spacy.load("en_core_web_md")  # 可以根据需要选择中文模型"zh_core_web_lg"
except:
    # 如果模型未安装，需要安装：python -m spacy download en_core_web_md
    print("spaCy模型未安装，部分功能可能不可用")
    nlp = None


class JobMatchingRL:
    """基于强化学习的职位匹配系统"""

    def __init__(self, learning_rate=0.1, discount_factor=0.8):
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.state_size = 10  # 状态空间大小
        self.action_size = 100  # 动作空间大小
        self.Q_table = np.zeros((self.state_size, self.action_size))  # Q表初始化
        self.scaler = StandardScaler()  # 状态标准化

    def extract_resume_info(self, resume_text):
        """从简历文本中提取关键信息"""
        if nlp is None:
            return {"skills": [], "education": "", "experience": ""}

        doc = nlp(resume_text)

        # 简单示例：提取技能（实际应用中需要更复杂的识别方法）
        skills = []
        skill_keywords = ['python', 'java', 'javascript', 'html', 'css', 'sql', 'react',
                          'angular', 'vue', 'django', 'flask', 'node', 'design',
                          'marketing', 'sales', 'communication', 'teamwork']

        for token in doc:
            if token.text.lower() in skill_keywords:
                skills.append(token.text.lower())

        # 提取教育和经验的简单逻辑
        education = ""
        experience = ""

        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(edu_term in sent_text for edu_term in
                   ['university', 'college', 'degree', 'bachelor', 'master', 'phd']):
                education += sent.text + " "
            if any(exp_term in sent_text for exp_term in ['worked', 'experience', 'job', 'position', 'role']):
                experience += sent.text + " "

        return {
            "skills": skills,
            "education": education.strip(),
            "experience": experience.strip()
        }

    def extract_job_info(self, job_description):
        """从职位描述中提取关键信息"""
        if nlp is None:
            return {"required_skills": [], "experience_required": "", "working_hours": []}

        doc = nlp(job_description)

        # 提取所需技能
        required_skills = []
        skill_keywords = ['python', 'java', 'javascript', 'html', 'css', 'sql', 'react',
                          'angular', 'vue', 'django', 'flask', 'node', 'design',
                          'marketing', 'sales', 'communication', 'teamwork']

        for token in doc:
            if token.text.lower() in skill_keywords:
                required_skills.append(token.text.lower())

        # 提取经验要求
        experience_required = ""
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if 'experience' in sent_text or 'year' in sent_text:
                experience_required += sent.text + " "

        # 提取工作时间
        working_hours = []
        time_patterns = ['full-time', 'part-time', 'morning', 'afternoon', 'evening', 'weekend']
        for pattern in time_patterns:
            if pattern in job_description.lower():
                working_hours.append(pattern)

        return {
            "required_skills": required_skills,
            "experience_required": experience_required.strip(),
            "working_hours": working_hours
        }

    def state_to_index(self, state):
        """将状态向量转换为索引"""
        # 简化版：取状态向量的平均值，映射到状态空间
        state_avg = np.mean(state)
        state_index = min(int(state_avg * self.state_size), self.state_size - 1)
        return max(0, state_index)  # 确保索引在有效范围内

    def get_state(self, user_preference, interaction_history):
        """将用户偏好和历史交互转换为状态向量"""
        # 提取用户偏好和历史交互的关键指标
        state_features = [
            float(user_preference.preferred_salary_range.get('min', 0)),
            float(user_preference.preferred_salary_range.get('max', 0)),
            len(user_preference.preferred_locations),
            len(user_preference.preferred_working_hours),
            len(interaction_history.get('viewed_jobs', [])),
            len(interaction_history.get('applied_jobs', [])),
            len(interaction_history.get('completed_jobs', [])),
            float(list(user_preference.skill_weights.values())[0]) if user_preference.skill_weights else 0.5,
            interaction_history.get('average_feedback', 0),
            interaction_history.get('application_success_rate', 0)
        ]

        # 对特征进行标准化处理
        if not hasattr(self, 'is_fitted') or not self.is_fitted:
            self.scaler.fit(np.array([state_features]))
            self.is_fitted = True

        normalized_state = self.scaler.transform([state_features])[0]
        return normalized_state

    def calculate_similarity(self, job, user_preference):
        """计算职位与用户偏好的相似度"""
        # 计算技能匹配度
        skill_match = self._calculate_skill_match(job, user_preference)

        # 计算位置匹配度
        location_match = self._calculate_location_match(job, user_preference)

        # 计算时间匹配度
        time_match = self._calculate_time_match(job, user_preference)

        # 计算薪资匹配度
        salary_match = self._calculate_salary_match(job, user_preference)

        # 获取权重
        weights = {
            'skill': 0.4,
            'location': 0.3,
            'time': 0.2,
            'salary': 0.1
        }

        # 计算总相似度 (PI - 个性化兴趣因子)
        pi = (weights['skill'] * skill_match +
              weights['location'] * location_match +
              weights['time'] * time_match +
              weights['salary'] * salary_match) / sum(weights.values())

        return pi

    def _calculate_skill_match(self, job, user_preference):
        """计算技能匹配度"""
        if not job.skills_required or not user_preference.skill_weights:
            return 0.5

        job_skills = json.loads(job.skills_required) if isinstance(job.skills_required, str) else job.skills_required
        user_skills = list(user_preference.skill_weights.keys())

        # 计算重叠技能比例
        if not job_skills or not user_skills:
            return 0.5

        # 将技能转为小写以进行不区分大小写的比较
        job_skills_lower = [s.lower() for s in job_skills]
        user_skills_lower = [s.lower() for s in user_skills]

        # 计算匹配的技能数
        matching_skills = set(job_skills_lower) & set(user_skills_lower)

        # 计算匹配比例
        job_match_ratio = len(matching_skills) / len(job_skills) if job_skills else 0
        user_match_ratio = len(matching_skills) / len(user_skills) if user_skills else 0

        # 取两者的加权平均
        return 0.6 * job_match_ratio + 0.4 * user_match_ratio

    def _calculate_location_match(self, job, user_preference):
        """计算位置匹配度"""
        if not job.location or not user_preference.preferred_locations:
            return 0.5

        job_location = job.location.lower()
        user_locations = [loc.lower() for loc in user_preference.preferred_locations]

        if job_location in user_locations:
            return 1.0

        # 如果位置不在用户偏好列表中，但用户接受远程工作
        if "remote" in user_locations and "remote" in job_location:
            return 0.9

        # 使用nlp进行位置相似度计算（如城市在同一地区）
        if nlp:
            max_similarity = 0
            job_loc_doc = nlp(job_location)

            for user_loc in user_locations:
                user_loc_doc = nlp(user_loc)
                similarity = job_loc_doc.similarity(user_loc_doc)
                max_similarity = max(max_similarity, similarity)

            return max_similarity

        return 0.1  # 默认较低的匹配度

    def _calculate_time_match(self, job, user_preference):
        """计算时间匹配度"""
        if not job.working_hours or not user_preference.preferred_working_hours:
            return 0.5

        job_hours = json.loads(job.working_hours) if isinstance(job.working_hours, str) else job.working_hours
        user_hours = user_preference.preferred_working_hours

        # 计算时间重叠比例
        if not job_hours or not user_hours:
            return 0.5

        # 将时间转为小写以进行不区分大小写的比较
        job_hours_lower = [t.lower() for t in job_hours]
        user_hours_lower = [t.lower() for t in user_hours]

        # 计算重叠时间段
        overlapping_hours = set(job_hours_lower) & set(user_hours_lower)

        # 计算重叠比例
        job_overlap_ratio = len(overlapping_hours) / len(job_hours) if job_hours else 0
        user_overlap_ratio = len(overlapping_hours) / len(user_hours) if user_hours else 0

        # 更偏向于满足工作要求
        return 0.7 * job_overlap_ratio + 0.3 * user_overlap_ratio

    def _calculate_salary_match(self, job, user_preference):
        """计算薪资匹配度"""
        if job.salary is None or not user_preference.preferred_salary_range:
            return 0.5

        # 将 decimal.Decimal 转换为 float 以确保类型兼容
        job_salary = float(job.salary)

        min_salary = float(user_preference.preferred_salary_range.get('min', 0))
        max_salary = float(user_preference.preferred_salary_range.get('max', float('inf')))

        # 如果薪资在用户期望范围内，完美匹配
        if min_salary <= job_salary <= max_salary:
            return 1.0

        # 如果薪资高于期望，也是好事
        if job_salary > max_salary:
            # 计算超出程度，最多不超过20%
            overage_ratio = min(1, max_salary / job_salary) if max_salary > 0 else 0
            return 0.8 + 0.2 * overage_ratio

        # 如果薪资低于期望
        if job_salary < min_salary and min_salary > 0:
            # 计算不足程度
            shortage_ratio = job_salary / min_salary
            return shortage_ratio

        return 0.5  # 默认中等匹配度

    def predict_user_actions(self, user, job, user_history):
        """预测用户对职位的不同行动概率"""
        # 计算基础匹配度
        base_similarity = self.calculate_similarity(job, user.userpreference)

        # 初始化默认概率
        probabilities = {
            'view': 0.6,  # 默认查看概率
            'save': 0.3,  # 默认保存概率
            'apply': 0.2,  # 默认申请概率
            'ignore': 0.3,  # 默认忽略概率
            'reject': 0.1,  # 默认拒绝概率
            'interview': 0.15,  # 默认面试概率
            'job_offer': 0.1,  # 默认录用概率
            'complete': 0.08,  # 默认完成工作概率
            'resign': 0.02  # 默认提前离职概率
        }

        # 根据匹配度调整概率
        probabilities['view'] = min(0.9, base_similarity * 1.2)
        probabilities['save'] = base_similarity * 0.7
        probabilities['apply'] = base_similarity * 0.5
        probabilities['ignore'] = max(0.1, (1 - base_similarity) * 0.5)
        probabilities['reject'] = max(0.05, (1 - base_similarity) * 0.3)

        # 根据历史数据调整概率
        if user_history:
            # 获取用户历史行为统计
            view_rate = user_history.get('view_rate', 0.5)
            apply_rate = user_history.get('apply_rate', 0.2)
            success_rate = user_history.get('success_rate', 0.1)

            # 调整概率
            probabilities['view'] = (probabilities['view'] + view_rate) / 2
            probabilities['apply'] = (probabilities['apply'] + apply_rate) / 2
            probabilities['interview'] = base_similarity * success_rate * 1.5
            probabilities['job_offer'] = base_similarity * success_rate
            probabilities['complete'] = base_similarity * success_rate * 0.8

        return probabilities

    def calculate_job_value(self, user, job, user_history):
        """计算职位对用户的总期望价值"""
        # 获取个性化兴趣因子
        pi = self.calculate_similarity(job, user.userpreference)

        # 预测用户行动概率
        probs = self.predict_user_actions(user, job, user_history)

        # 计算即时奖励期望值
        v_immediate = (probs['view'] * 0.1) + (probs['save'] * 0.3) + (probs['apply'] * 0.5) - \
                      (probs['ignore'] * 0.1) - (probs['reject'] * 0.3)

        # 计算延迟奖励期望值
        v_delayed = (probs['interview'] * 1.0) + (probs['job_offer'] * 2.0) + \
                    (probs['complete'] * 3.0) - (probs['resign'] * 1.0)

        # 总价值计算
        discount_factor = 0.8  # 延迟奖励折扣因子
        total_value = pi * (v_immediate + v_delayed * discount_factor)

        return total_value

    def recommend_jobs(self, user, available_jobs, num_recommendations=5):
        """推荐最适合用户的职位"""
        # 获取用户偏好
        try:
            user_preference = UserPreference.objects.get(user=user)
            user_history = user_preference.interaction_history or {}
        except UserPreference.DoesNotExist:
            # 如果用户没有设置偏好，返回空列表
            return []

        # 计算每个职位的匹配值
        job_scores = []
        for job in available_jobs:
            # 获取当前状态
            current_state = self.get_state(user_preference, user_history)
            state_index = self.state_to_index(current_state)

            # 获取职位对应的动作索引
            action_index = hash(str(job.id)) % self.action_size

            # 获取Q值
            q_value = self.Q_table[state_index][action_index]

            # 计算职位相似度
            similarity = self.calculate_similarity(job, user_preference)

            # 计算职位总价值
            job_value = self.calculate_job_value(user, job, user_history)

            # 结合Q值和相似度，计算最终分数
            # 新职位Q值为0，主要依赖相似度；有交互历史的职位会更多考虑Q值
            exploration_factor = 0.2  # 探索因子
            if q_value > 0:
                final_score = (1 - exploration_factor) * (
                            0.7 * q_value + 0.3 * job_value) + exploration_factor * similarity
            else:
                final_score = (1 - exploration_factor) * job_value + exploration_factor * similarity

            job_scores.append((job, final_score))

        # 按分数排序
        job_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回推荐结果
        return [job for job, score in job_scores[:num_recommendations]]

    def update_q_value(self, user, job, action, reward):
        """基于用户反馈更新Q值"""
        try:
            user_preference = UserPreference.objects.get(user=user)
            user_history = user_preference.interaction_history or {}
        except UserPreference.DoesNotExist:
            return

        # 获取当前状态
        current_state = self.get_state(user_preference, user_history)
        current_state_index = self.state_to_index(current_state)

        # 获取职位对应的动作索引
        action_index = hash(str(job.id)) % self.action_size

        # 更新Q值
        current_q = self.Q_table[current_state_index][action_index]

        # 简化版：不计算下一状态的最大Q值，直接使用即时奖励
        # 在实际应用中，应该计算下一状态的最大Q值
        new_q = current_q + self.learning_rate * (reward - current_q)

        # 更新Q表
        self.Q_table[current_state_index][action_index] = new_q

        return new_q


# 全局实例
job_matcher = JobMatchingRL()


# 视图函数
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